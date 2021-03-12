from urllib.parse import urlparse

import allrank.models.losses as losses
import numpy as np
import os
import torch
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel
from allrank.training.train_utils import fit, fit_with_distillation
from allrank.utils.command_executor import execute_command
from allrank.utils.experiments import dump_experiment_result, assert_expected_metrics
from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr
from allrank.training.train_utils import compute_metrics
from argparse import ArgumentParser, Namespace
from attr import asdict
from functools import partial
from pprint import pformat
from torch import optim
#from allrank.training import optimizers


import sys

def parse_args() -> Namespace:
    parser = ArgumentParser("allRank")
    parser.add_argument("--job-dir", help="Base output path for all experiments", required=True)
    parser.add_argument("--run-id", help="Name of this run to be recorded (must be unique within output dir)",
                        required=True)
    parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with config")
    parser.add_argument("--evaluate", action="store_true", default=False)
    return parser.parse_args()


def run(args):
    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)


    paths = PathsContainer.from_args(args.job_dir, args.run_id, args.config_file_name)

    create_output_dirs(paths.output_dir)

    logger = init_logger(paths.output_dir)
    logger.info(f"created paths container {paths}")

    # read config
    config = Config.from_json(paths.config_path)
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

    output_config_path = os.path.join(paths.output_dir, "used_config.json")
    execute_command("cp {} {}".format(paths.config_path, output_config_path))

    print("Shared in main", config.data.shared)
    # train_ds, val_ds, test_ds
    train_ds, val_ds, test_ds = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
        test_ds_role=config.data.test_ds_role,
        sigma=config.data.noise,
        shared=config.data.shared

    )

    n_features = train_ds.shape[-1]
    assert n_features == val_ds.shape[-1], "Last dimensions of train_ds and val_ds do not match!"

    # train_dl, val_dl, test_dl
    train_dl, val_dl, test_dl  = create_data_loaders(
        train_ds, val_ds, test_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)

    # gpu support
    dev = get_torch_device()
    logger.info("Model training will execute on {}".format(dev.type))

    # instantiate model

    use_distillation = True if config.distillation_loss else False
    full_pipeline = True if use_distillation and "full" in config.distillation_loss.name else False
    fit_size = config.teacher_model.fc_model['sizes'][-1] if full_pipeline and config.teacher_model.fc_model['sizes'][-1] != config.model.fc_model['sizes'][-1]  else None
    print("Fit size", fit_size)
    model = make_model(n_features=n_features, **asdict(config.model, recurse=False), fit_size=fit_size, distillation=full_pipeline, seq_len= config.data.slate_length)
    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
        logger.info("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
    model.to(dev)

    # load optimizer, loss and LR scheduler
    if hasattr(optim, config.optimizer.name):
        optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)
    #if hasattr(optimizers, config.optimizer.name):
    #    optimizer = getattr(optimizers, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)
    if config.lr_scheduler.name:
        scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, **config.lr_scheduler.args)
    else:
        scheduler = None
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)

    if args.evaluate:
        test_metrics = compute_metrics(config.metrics, model, test_dl, dev)
        print(test_metrics)
        sys.exit()

    if use_distillation:
        if full_pipeline:
            assert config.teacher_model.transformer.h == config.model.transformer.h
        teacher_model = make_model(n_features=n_features, **asdict(config.teacher_model, recurse=False),
                                   distillation=full_pipeline, fit_size=None)
        if torch.cuda.device_count() > 1:
            teacher_model = CustomDataParallel(teacher_model)
            logger.info("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
        teacher_model.to(dev)
        loss_func = partial(getattr(losses, config.distillation_loss.name), gt_loss_func=loss_func,
                            **config.distillation_loss.args)
        with torch.autograd.detect_anomaly() if config.detect_anomaly else dummy_context_mgr():  # type: ignore
            result, model = fit_with_distillation(
                student_model=model,
                teacher_model=teacher_model,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                train_dl=train_dl,
                valid_dl=val_dl,
                config=config,
                device=dev,
                output_dir=paths.output_dir,
                tensorboard_output_path=paths.tensorboard_output_path,
                full=full_pipeline,
                **asdict(config.training)
            )

    else:
        with torch.autograd.detect_anomaly() if config.detect_anomaly else dummy_context_mgr():  # type: ignore
            # run training
            result, model = fit(
                model=model,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                train_dl=train_dl,
                valid_dl=val_dl,
                config=config,
                device=dev,
                output_dir=paths.output_dir,
                tensorboard_output_path=paths.tensorboard_output_path,
                **asdict(config.training)
            )
    #Reload best model
    sd = torch.load(os.path.join(paths.output_dir, "best_model.pkl"))
    model.load_state_dict(sd)
    test_metrics = compute_metrics(config.metrics, model, test_dl, dev)
    result['test_metrics'] = test_metrics
    print(result)
    dump_experiment_result(args, config, paths.output_dir, result)

    if urlparse(args.job_dir).scheme == "gs":
        copy_local_to_gs(paths.local_base_output_path, args.job_dir)

    assert_expected_metrics(result, config.expected_metrics)
    return test_metrics['ndcg_10']

if __name__ == "__main__":
    args = parse_args()

    run(args)
