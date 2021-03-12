import os
import sys
import json


def find_best_res_for_architecture(architecture_folder_path):
    best_result = 0.
    for exp in os.listdir(architecture_folder_path):
        current_exp_path = os.path.join(architecture_folder_path, exp)
        if os.path.isdir(current_exp_path):
            with open(os.path.join(current_exp_path, "experiment_result.json")) as f:
                j_file = json.load(f)
            current_result = j_file['test_metrics/ndcg_10']
            if current_result > best_result:
                best_result = current_result
                best_model = exp

    return best_result, best_model

def main():

    res_folder = sys.argv[1]
    if len(sys.argv) > 2:
        architecture = sys.argv[2]
    else:
        architecture = None
    if architecture is None:
        for architecture_name in os.listdir(res_folder):
            architecture_folder_path = os.path.join(res_folder, architecture_name)
            if os.path.isdir(architecture_folder_path):
                best_result, best_model = find_best_res_for_architecture(architecture_folder_path)
                print(architecture_folder_path, best_result, best_model )
    else:
        best_result, best_model = find_best_res_for_architecture(architecture)
        print(architecture, best_result, best_model)


if __name__=="__main__":
    main()






