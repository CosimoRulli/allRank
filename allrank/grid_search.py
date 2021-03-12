## Grid Search for Ordinal Loss
from sklearn.model_selection import ParameterGrid
import os
import json
from allrank.main import run, parse_args
import pandas as pd
import numpy as np

def write_configuration_on_json(json_path, conf):
    with open(json_path) as f:
        j_file = json.load(f)

    if "sizes_fc" in conf:
        j_file["model"]["fc_model"]["sizes"] = (conf["sizes_fc"])
    if "dropout_fc" in conf:
        j_file["model"]["fc_model"]["dropout"] = conf["dropout_fc"]
    if "N" in conf:
        j_file["model"]["transformer"]["N"] = conf["N"]
    if "h" in conf:
        j_file["model"]["transformer"]["h"] = conf["h"]
    if "d_ff" in conf:
        j_file["model"]["transformer"]["d_ff"] = conf["d_ff"]
    if "dropout" in conf:
        j_file["model"]["transformer"]["dropout"] = conf["dropout"]
    if "batch_size" in conf:
        j_file["data"]["batch_size"] = conf["batch_size"]
    if "noise" in conf:
        j_file["data"]["noise"] = conf["noise"]
    if "lr" in conf:
        j_file["optimizer"]["args"]["lr"] = conf["lr"]
    if "weight_decay" in conf:
        j_file["optimizer"]["args"]["weight_decay"] = conf["weight_decay"]
    if "gamma" in conf:
        j_file["lr_scheduler"]["args"]["gamma"] = conf["gamma"]
    if "alpha" in conf:
        j_file["distillation_loss"]["args"]["alpha"] = conf["alpha"]
    if "temperature" in conf:
        j_file["distillation_loss"]["args"]["temperature"] = conf["temperature"]

    with open(json_path, "w") as f:
        json.dump(j_file, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    name_of_experiment = args.run_id
    base_log_dir = args.job_dir
    json_path = args.config_file_name
    base_log_dir = os.path.join(base_log_dir, name_of_experiment)
    print(json_path)

    params = {}


    #params["sizes_fc"] = [[144]]
    #params["N"] = [5]
    #params["h"] = [2,4]
    #params["d_ff"] = [512, 1024]

    training_params = {}
    training_params['batch_size'] = [32]
    #training_params["dropout_fc"] = [0.]
    training_params["dropout"] = [0.0, 0.1 ]
    training_params['lr'] = [0.005, 0.001]
    alphas = list(np.linspace(0.1, 1, 10, endpoint=True))
    alphas.reverse()
    #training_params["alpha"] = [0.5, 1.0, 5]
    training_params["weight_decay"] = [0.0, 1e-6, 1e-5]
    training_params["gamma"] = [0.5, 0.1]
    #training_params["noise"] = [0.5, 1.0]

    columns = list(params.keys())
    columns.append("best_metric")
    dataframe_results = pd.DataFrame(columns=columns)


    for param_conf_architecture in ParameterGrid(params):
        log_dir_name = ""
        for key in params.keys():
            log_dir_name += "_" + key.replace("_", "") + "_" + str(param_conf_architecture[key])
        log_dir_name = log_dir_name[1:]
        current_architecture_log_dir = os.path.join(base_log_dir, log_dir_name)

        best_result = 0
        for training_conf in ParameterGrid(training_params):

            training_configuration_name = ""
            for key in training_conf.keys():
                training_configuration_name += "_" + key.replace("_", "") + "_" + str(training_conf[key])
            training_configuration_name = training_configuration_name[1:].replace("[", "").replace("]", "")

            final_log_dir = os.path.join(current_architecture_log_dir, training_configuration_name)

            final_conf = dict(param_conf_architecture, **training_conf)

            write_configuration_on_json(json_path, final_conf)

            args.run_id = final_log_dir

            test_ndcg = run(args)
            if test_ndcg > best_result:
                best_result = test_ndcg

            print(final_conf)
            print(test_ndcg)
        new_line = list(param_conf_architecture)
        new_line.append(best_result)
        current_df = pd.DataFrame([new_line], columns=columns)
        dataframe_results = dataframe_results.append(current_df)

    dataframe_results.to_csv(os.path.join("results",base_log_dir, "result_summary.csv"))

