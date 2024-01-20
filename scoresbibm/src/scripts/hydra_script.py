
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

import torch
import jax
import numpy as np
import random

import os 
import sys
import socket
import time
from scoresbibm.src.evaluation.eval_task import eval_unstructured_task

from scoresbibm.src.tasks import get_task
from scoresbibm.src.methods.method_base import get_method
from scoresbibm.src.evaluation import get_metric, eval_inference_task, eval_all_conditional_task
from scoresbibm.src.tasks.base_task import AllConditionalTask, InferenceTask
from scoresbibm.src.tasks.unstructured_tasks import UnstructuredTask
from scoresbibm.src.utils.data_utils import init_dir, generate_unique_model_id, save_model, save_summary







ascii_logo = """
   _____                     _____ ____ _____ 
  / ____|                   / ____|  _ \_   _|
 | (___   ___ ___  _ __ ___| (___ | |_) || |  
  \___ \ / __/ _ \| '__/ _ \\___ \|  _ < | |  
  ____) | (_| (_) | | |  __/____) | |_) || |_ 
 |_____/ \___\___/|_|  \___|_____/|____/_____|
"""


def main():
    """ Main script to run"""
    print(ascii_logo)
    score_sbi()
    
    
@hydra.main(version_base=None, config_path="../../config", config_name="config.yaml")
def score_sbi(cfg: DictConfig):
    """Evaluate score based inference"""
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Go back to the folder named "cfg.name"
    output_super_dir = os.path.dirname(output_dir)
    while os.path.basename(output_super_dir) != cfg.name:
        output_super_dir = os.path.dirname(output_super_dir)

    log.info(f"Working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")
    log.info("Output super directory: {}".format(output_super_dir))
    log.info(f"Hostname: {socket.gethostname()}")
    log.info(f"Jax devices: {jax.devices()}")
    log.info(f"Torch devices: {torch.cuda.device_count()}")
    
    seed = cfg.seed
    rng = set_seed(seed)    
    backend = cfg.method.backend
    log.info(f"Seed: {seed}")
    
    init_dir(output_super_dir)
    
    # Set up the task # TODO: maybe add data_device
    log.info(f"Task: {cfg.task.name}")
    task = get_task(cfg.task.name, backend=backend)
    data = task.get_data(cfg.task.num_simulations, rng=rng)

    # Run method
    log.info(f"Running method: {cfg.method.name}")
    method_run = get_method(cfg.method.name)
    rng, rng_train = jax.random.split(rng)
    start_time = time.time()
    model = method_run(task,data, cfg.method, rng=rng_train)
    time_train = time.time() - start_time
    log.info(f"Training time: {time_train}")
    
    del data 

    # Evaluate
    log.info(f"Evaluating method: {cfg.method.name}")
    metrics = cfg.eval["metric"]
    metrics_results = {}
    for m, metric_params in metrics.items():
        log.info(f"Evaluating metric: {m}")
        rng, rng_eval = jax.random.split(rng)
        if m == "none":
            continue
        metric_fn = get_metric(str(m))
        
        if issubclass(type(task), InferenceTask):
            metric_values, eval_time = eval_inference_task(task, model, metric_fn, metric_params, rng_eval)
        elif issubclass(task.__class__, UnstructuredTask):
            metric_values, eval_time = eval_unstructured_task(task, model, metric_fn, metric_params, rng_eval)
        elif issubclass(task.__class__, AllConditionalTask):
            metric_values, eval_time = eval_all_conditional_task(task, model, metric_fn, metric_params, rng_eval)
        else:
            raise ValueError("Task not recognized.")
        
        if metric_values is not None:
            metrics_results[m] = metric_values
        
    if len(metrics_results) == 0:
        # To get a summary entry for the model
        metrics_results["none"] = None
        eval_time = None
        
    # Saving results
    is_save_model = cfg.save_model
    if is_save_model:
        log.info(f"Saving model")
        model_id = generate_unique_model_id(output_super_dir)
        try:
            save_model(model, output_super_dir, model_id)
            log.info(f"Model saved with id: {model_id}")
        except Exception as e:
            log.info("Tried to save model, but failed.")
            log.info(e)
            
        
    # Save summary
    is_save_summary = cfg.save_summary
    if is_save_summary:
        log.info(f"Saving summary")
        model_id = generate_unique_model_id(output_super_dir)
        try:
            for m, vals in metrics_results.items():
                save_summary(output_super_dir, cfg.method.name, cfg.task.name, cfg.task.num_simulations, model_id, m, vals, seed, time_train, eval_time, cfg)
        except Exception as e:
            log.info("Tried to save summary, but failed.")
            log.info(e)
        log.info(f"Summary saved with id: {model_id}")
        
        
    if cfg.sweeper.name is not None:
        objective_sweep = cfg.sweeper.objective
        # Return average metric value for sweeps    
        return sum(metrics_results[objective_sweep]) / len(metrics_results[objective_sweep])
    else:
        return 0.
    
        
    
    
    
def set_seed(seed:int):
    """This methods just sets the seed."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(seed)
    return key
