
import torch
import jax 
import jax.numpy as jnp

import numpy as np

import os
import pandas as pd

import pickle

def init_dir(dir_path):
    if not os.path.exists(dir_path + os.sep + "models"):
        os.makedirs(dir_path + os.sep + "models")
        
    if not os.path.exists(dir_path + os.sep + "summary.csv"):
        df = pd.DataFrame(columns=["method", "task", "num_simulations", "seed",  "model_id", "metric", "value", "cfg"])
        df.to_csv(dir_path + os.sep + "summary.csv", index=False)
        
        
def get_summary_df(dir_path):
    return pd.read_csv(dir_path + os.sep + "summary.csv")

def generate_unique_model_id(dir_path):
    summary_df = get_summary_df(dir_path)
    return len(summary_df)
        
        
def save_model(model, dir_path, model_id):
    file_name = dir_path + os.sep + "models" + os.sep + f"model_{model_id}.pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)
        
def save_summary(dir_path, method:str, task:str, num_simulations:int, model_id:int, metric:str, value:float, seed:int, cfg:dict):
    summary_df = get_summary_df(dir_path)
    new_row = pd.DataFrame({"method": method, "task": task, "num_simulations": num_simulations, "seed": seed, "model_id": model_id, "metric": metric, "value": str(value), "cfg": str(cfg)}, index=[len(summary_df)])
    summary_df = pd.concat([summary_df, new_row], axis=0, ignore_index=True)
    summary_df.to_csv(dir_path + os.sep + "summary.csv", index=False)

def load_model(dir_path, model_id):
    file_name = dir_path + os.sep + "models" + os.sep + f"{model_id}.pkl"
    with open(file_name, 'rb') as file:
        return pickle.load(file)
        


def as_torch_tensor(*args):
    updated_args = [torch.from_numpy(np.asarray(a)) for a in args]
    return updated_args

def as_jax_array(*args):
    updated_args = [jnp.asarray(np.asarray(a)) for a in args]
    return updated_args
    