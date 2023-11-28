
import torch
import jax 
import jax.numpy as jnp

import numpy as np
import math

import os
import pandas as pd

import pickle

def init_dir(dir_path):
    if not os.path.exists(dir_path + os.sep + "models"):
        os.makedirs(dir_path + os.sep + "models")
        
    if not os.path.exists(dir_path + os.sep + "summary.csv"):
        df = pd.DataFrame(columns=["method", "task", "num_simulations", "seed",  "model_id", "metric", "value", "cfg"])
        df.to_csv(dir_path + os.sep + "summary.csv", index=False)
        
        
        
def query(name,task=None,method = None, num_simulations=None, metric = None, seed = None, value_statistic = "mean", **kwargs):
    summary_df = get_summary_df(name)
    query = to_query_string("method", method)
    if num_simulations is not None:
        if query != "":
            query += "&"
        query +=  to_query_string("num_simulations", num_simulations)
    elif seed is not None:
        if query != "":
            query += "&"
        query += to_query_string("seed", seed)
    elif task is not None:
        if query != "":
            query += "&"
        query +=  to_query_string("task", task)
    elif metric is not None:
        if query != "":
            query += "&"
        query += to_query_string("metric", metric)

    print(query)

    if query == "":
        df_q = summary_df
    else:
        df_q = summary_df.query(query)
    
    # Evaluate value, which is a string
    df_q["value"] = df_q["value"].apply(lambda x: np.array(eval(x)))
    if value_statistic == "mean":
        df_q["value"] = df_q["value"].apply(lambda x: np.mean(x))
    elif value_statistic == "median":
        df_q["value"] = df_q["value"].apply(lambda x: np.median(x))
    elif value_statistic == "std":
        df_q["value"] = df_q["value"].apply(lambda x: np.std(x))
    elif "quantile" in value_statistic:
        val = float(value_statistic.split("_")[1])
        df_q["value"] = df_q["value"].apply(lambda x: np.quantile(x, val))
    else:
        raise NotImplementedError()
    return df_q
        
    



def to_query_string(name: str, var) -> str:
    """Translates a variable to string.

    Args:
        name (str): Query argument
        var (str): value

    Returns:
        str: Query == value ?
    """
    if var is None:
        return ""
    elif var is pd.NA or var is torch.nan or var is math.nan or str(var) == "nan" or var is jnp.nan:
        return f"{name}!={name}"
    elif isinstance(var, list) or isinstance(var, tuple):
        query = "("
        for v in var:
            if query != "(":
                query += "|"
            if isinstance(v, str):
                query += f"{name}=='{v}'"
            else:
                query += f"{name}=={v}"
        query += ")"
    else:
        if isinstance(var, str):
            query = f"{name}=='{var}'"
        else:
            query = f"{name}=={var}"
    return query
  
        
def get_summary_df(dir_path):
    df = pd.read_csv(dir_path + os.sep + "summary.csv")
    return df 


def generate_unique_model_id(dir_path):
    summary_df = get_summary_df(dir_path)
    model_ids = summary_df["model_id"].values
    if len(model_ids) == 0:
        return 0
    elif len(model_ids) == 1:
        return 1
    else:
        max_id = np.max(model_ids)
        return max_id + 1
        
        
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
    