import os
import shutil
import numpy as np
import torch
import sys


if __name__=="__main__":
    folder = sys.argv[1]
    sd = torch.load(os.path.join(folder, "model.pkl"), map_location="cpu")
    folder = os.path.join(folder, "numpy_weights")
    if not os.path.exists(folder):
        os.mkdir(folder)
    for key, item in sd.items():
        print(key)
        model_name, layer_name = key.split(".", maxsplit=1)
        model_folder = os.path.join(folder, model_name)
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        if model_name != "encoder" or layer_name.split(".", maxsplit=1)[0] == "norm":
            item_path = os.path.join(model_folder, layer_name)
            item = torch.where(torch.abs(item) > 1e-8, item, torch.zeros(item.shape)).numpy()
            np.save(item_path, item)
        else:
            _, layer_number, layer_name = layer_name.split(".", maxsplit=2)
            current_layer_dir = os.path.join(model_folder, "layer_"+ layer_number)
            if not os.path.exists(current_layer_dir):
                os.mkdir(current_layer_dir)
            item_path = os.path.join(current_layer_dir, layer_name)
            item = torch.where(torch.abs(item) > 1e-8, item, torch.zeros(item.shape)).numpy()
            np.save(item_path, item)



