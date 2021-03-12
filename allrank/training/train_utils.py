import os
from functools import partial

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

import allrank.models.metrics as metrics_module
from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.model_utils import get_num_params, log_num_params
from allrank.training.early_stop import EarlyStop
from allrank.utils.ltr_logging import get_logger
from allrank.utils.tensorboard_utils import TensorboardSummaryWriter

logger = get_logger()


from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR


def loss_batch(model, loss_func, xb, yb, indices, gradient_clipping_norm, opt=None):
    mask = (yb == PADDED_Y_VALUE)
    loss = loss_func(model(xb, mask, indices), yb)

    if opt is not None:
        loss.backward()
        if gradient_clipping_norm:
            clip_grad_norm_(model.parameters(), gradient_clipping_norm)
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def loss_batch_swa(model, loss_func, xb, yb, indices, gradient_clipping_norm, opt=None, swa_model=None):
    mask = (yb == PADDED_Y_VALUE)
    loss = loss_func(model(xb, mask, indices), yb)

    if opt is not None:
        loss.backward()
        if gradient_clipping_norm:
            clip_grad_norm_(model.parameters(), gradient_clipping_norm)
        opt.step()
        swa_model.update_parameters(model)
        opt.zero_grad()

    return loss.item(), len(xb)



def loss_batch_distillation(student_model, teacher_model, loss_func, xb, yb, indices, gradient_clipping_norm, opt=None):
    mask = (yb == PADDED_Y_VALUE)
    with torch.no_grad():
        teacher_out = teacher_model(xb, mask, indices)
    loss = loss_func(student_model(xb, mask, indices), teacher_out, yb)

    if opt is not None:
        loss.backward()
        if gradient_clipping_norm:
            clip_grad_norm_(student_model.parameters(), gradient_clipping_norm)
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def loss_batch_distillation_full(student_model, teacher_model, loss_func, xb, yb, indices, gradient_clipping_norm, opt=None):
    mask = (yb == PADDED_Y_VALUE)
    with torch.no_grad():
        teacher_pred, teacher_h, teacher_attn = teacher_model(xb, mask, indices)
    stud_pred, stud_h, stud_attn = student_model(xb, mask, indices)
    loss = loss_func(stud_pred, stud_h, stud_attn, teacher_pred, teacher_h, teacher_attn, yb)
    if opt is not None:
        loss.backward()
        if gradient_clipping_norm:
            clip_grad_norm_(student_model.parameters(), gradient_clipping_norm)
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def metric_on_batch(metric, model, xb, yb, indices):
    mask = (yb == PADDED_Y_VALUE)
    return metric(model.score(xb, mask, indices), yb)


def metric_on_epoch(metric, model, dl, dev):
    metric_values = torch.mean(
        torch.cat(
            [metric_on_batch(metric, model, xb.to(device=dev), yb.to(device=dev), indices.to(device=dev))
             for xb, yb, indices in dl]
        ), dim=0
    ).cpu().numpy()
    return metric_values


def compute_metrics(metrics, model, dl, dev):
    metric_values_dict = {}
    for metric_name, ats in metrics.items():
        metric_func = getattr(metrics_module, metric_name)
        metric_func_with_ats = partial(metric_func, ats=ats)
        metrics_values = metric_on_epoch(metric_func_with_ats, model, dl, dev)
        metrics_names = ["{metric_name}_{at}".format(metric_name=metric_name, at=at) for at in ats]
        metric_values_dict.update(dict(zip(metrics_names, metrics_values)))

    return metric_values_dict


def epoch_summary(epoch, train_loss, val_loss, train_metrics, val_metrics):
    summary = "Epoch : {epoch} Train loss: {train_loss} Val loss: {val_loss}".format(
        epoch=epoch, train_loss=train_loss, val_loss=val_loss)
    for metric_name, metric_value in train_metrics.items():
        summary += " Train {metric_name} {metric_value}".format(
            metric_name=metric_name, metric_value=metric_value)

    for metric_name, metric_value in val_metrics.items():
        summary += " Val {metric_name} {metric_value}".format(
            metric_name=metric_name, metric_value=metric_value)

    return summary


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit(epochs, model, loss_func, optimizer, scheduler, train_dl, valid_dl, config,
        gradient_clipping_norm, early_stopping_patience, device, output_dir, tensorboard_output_path):
    tensorboard_summary_writer = TensorboardSummaryWriter(tensorboard_output_path)


    num_params = get_num_params(model)
    log_num_params(num_params)

    early_stop = EarlyStop(early_stopping_patience)
    best_metric = -1
    best_model = None

    for epoch in range(epochs):
        logger.info("Current learning rate: {}".format(get_current_lr(optimizer)))

        model.train()
        # xb dim: [batch_size, slate_length, embedding_dim]
        # yb dim: [batch_size, slate_length]

        train_losses, train_nums = zip(
            *[loss_batch(model, loss_func, xb.to(device=device), yb.to(device=device), indices.to(device=device),
                         gradient_clipping_norm, optimizer) for
              xb, yb, indices in train_dl])
        train_loss = np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
        train_metrics = compute_metrics(config.metrics, model, train_dl, device)

        model.eval()
        with torch.no_grad():
            val_losses, val_nums = zip(
                *[loss_batch(model, loss_func, xb.to(device=device), yb.to(device=device), indices.to(device=device),
                             gradient_clipping_norm) for
                  xb, yb, indices in valid_dl])
            val_metrics = compute_metrics(config.metrics, model, valid_dl, device)

        val_loss = np.sum(np.multiply(val_losses, val_nums)) / np.sum(val_nums)

        tensorboard_metrics_dict = {("train", "loss"): train_loss, ("val", "loss"): val_loss}

        train_metrics_to_tb = {("train", name): value for name, value in train_metrics.items()}
        tensorboard_metrics_dict.update(train_metrics_to_tb)
        val_metrics_to_tb = {("val", name): value for name, value in val_metrics.items()}
        tensorboard_metrics_dict.update(val_metrics_to_tb)
        tensorboard_metrics_dict.update({("train", "lr"): get_current_lr(optimizer)})

        tensorboard_summary_writer.save_to_tensorboard(tensorboard_metrics_dict, epoch)

        logger.info(epoch_summary(epoch, train_loss, val_loss, train_metrics, val_metrics))

        current_val_metric_value = val_metrics.get(config.val_metric)
        if current_val_metric_value > best_metric:
            best_metric = current_val_metric_value
            best_model = model
            torch.save(best_model.state_dict(), os.path.join(output_dir, "best_model.pkl"))
        if scheduler:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                args = [val_metrics[config.val_metric]]
                scheduler.step(*args)
            else:
                scheduler.step()

        early_stop.step(current_val_metric_value, epoch)
        if early_stop.stop_training(epoch):
            logger.info(
                "early stopping at epoch {} since {} didn't improve from epoch no {}. Best value {}, current value {}".format(
                    epoch, config.val_metric, early_stop.best_epoch, early_stop.best_value, current_val_metric_value
                ))
            break

    torch.save(best_model.state_dict(), os.path.join(output_dir, "last_checkpoint.pkl"))
    tensorboard_summary_writer.close_all_writers()

    return {
        "epochs": epoch,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "num_params": num_params
    }, best_model


def fit_with_distillation(epochs, student_model, teacher_model, loss_func, optimizer, scheduler, train_dl, valid_dl,
                          config,
                          gradient_clipping_norm, early_stopping_patience, device, output_dir, tensorboard_output_path, full):
    tensorboard_summary_writer = TensorboardSummaryWriter(tensorboard_output_path)

    loss_batch_distillation_fun = loss_batch_distillation_full if full else loss_batch_distillation
    logger.info("Training with distillation")

    num_params = get_num_params(student_model)
    log_num_params(num_params)

    early_stop = EarlyStop(early_stopping_patience)
    best_metric = -1
    best_model = None
    teacher_model.eval()

    for epoch in range(epochs):
        logger.info("Current learning rate: {}".format(get_current_lr(optimizer)))

        student_model.train()

        # xb dim: [batch_size, slate_length, embedding_dim]
        # yb dim: [batch_size, slate_length]

        train_losses, train_nums = zip(
            *[loss_batch_distillation_fun(student_model, teacher_model, loss_func, xb.to(device=device), yb.to(device=device), indices.to(device=device),
                         gradient_clipping_norm, optimizer) for
              xb, yb, indices in train_dl])
        train_loss = np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
        train_metrics = compute_metrics(config.metrics, student_model, train_dl, device)

        student_model.eval()
        with torch.no_grad():
            val_losses, val_nums = zip(
                *[loss_batch_distillation_fun(student_model, teacher_model, loss_func, xb.to(device=device), yb.to(device=device), indices.to(device=device),
                             gradient_clipping_norm) for
                  xb, yb, indices in valid_dl])
            val_metrics = compute_metrics(config.metrics, student_model, valid_dl, device)

        val_loss = np.sum(np.multiply(val_losses, val_nums)) / np.sum(val_nums)

        tensorboard_metrics_dict = {("train", "loss"): train_loss, ("val", "loss"): val_loss}

        train_metrics_to_tb = {("train", name): value for name, value in train_metrics.items()}
        tensorboard_metrics_dict.update(train_metrics_to_tb)
        val_metrics_to_tb = {("val", name): value for name, value in val_metrics.items()}
        tensorboard_metrics_dict.update(val_metrics_to_tb)
        tensorboard_metrics_dict.update({("train", "lr"): get_current_lr(optimizer)})

        tensorboard_summary_writer.save_to_tensorboard(tensorboard_metrics_dict, epoch)

        logger.info(epoch_summary(epoch, train_loss, val_loss, train_metrics, val_metrics))

        current_val_metric_value = val_metrics.get(config.val_metric)
        #print("Current val metric value", current_val_metric_value)
        #print(output_dir)
        if current_val_metric_value > best_metric:
            best_model = student_model
            best_metric = current_val_metric_value
            torch.save(student_model.state_dict(), os.path.join(output_dir, "best_model.pkl"))

        if scheduler:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                args = [val_metrics[config.val_metric]]
                scheduler.step(*args)
            else:
                scheduler.step()

        early_stop.step(current_val_metric_value, epoch)
        if early_stop.stop_training(epoch):
            logger.info(
                "early stopping at epoch {} since {} didn't improve from epoch no {}. Best value {}, current value {}".format(
                    epoch, config.val_metric, early_stop.best_epoch, early_stop.best_value, current_val_metric_value
                ))
            break

    torch.save(best_model.state_dict(), os.path.join(output_dir, "last_checkpoint.pkl"))
    tensorboard_summary_writer.close_all_writers()

    return {
        "epochs": epoch,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "num_params": num_params
    }, best_model
