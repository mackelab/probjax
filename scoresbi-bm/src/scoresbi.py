
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

import torch
import numpy as np
import random

import os 
import sys
import socket

from tasks import get_task
from methods import get_method
from eval import get_metric
from utils import init_dir, generate_unique_model_id, save_model, save_summary





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
    
    
@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def score_sbi(cfg: DictConfig):
    """Evaluate score based inference"""
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    output_super_dir = os.path.dirname(os.path.dirname(output_dir))

    log.info(f"Working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")
    log.info(f"Hostname: {socket.gethostname()}")
    
    seed = cfg.seed
    set_seed(seed)    
    log.info(f"Seed: {seed}")
    
    
    init_dir(output_super_dir)
    
    # Set up the task
    log.info(f"Task: {cfg.task.name}")
    task = get_task(cfg.task.name, backend="torch")
    thetas, xs = task.get_thetas_xs(cfg.task.num_simulations)
    
    
    # Run method
    log.info(f"Running method: {cfg.method.name}")
    method_run = get_method(cfg.method.name)
    model = method_run(task,thetas, xs, cfg.method)
    
    # Evaluate
    log.info(f"Evaluating method: {cfg.method.name}")
    metrics = cfg.eval["metric"]
    metrics_results = {}
    for m, metric_params in metrics.items():
        log.info(f"Evaluating metric: {m}")
        metric_fn = get_metric(str(m))
        metric_values = []
        for i in task.observations:
            x_o = task.get_observation(i)
            true_posterior_samples = task.get_reference_posterior_samples(i)
            est_posterior_samples = model.sample(num_samples=true_posterior_samples.shape[0], x_o=x_o)
            
            val = metric_fn(true_posterior_samples, est_posterior_samples, **metric_params)
            metric_values.append(val)
            
        log.info(f"Metric values: {metric_values}")
        metrics_results[m] = metric_values
            
            
    # Saving results
    is_save_model = cfg.save_model
    if is_save_model:
        log.info(f"Saving model")
        model_id = generate_unique_model_id(output_super_dir)
        save_model(model, output_super_dir, model_id)
        log.info(f"Model saved with id: {model_id}")
        
    # Save summary
    is_save_summary = cfg.save_summary
    if is_save_summary:
        log.info(f"Saving summary")
        model_id = generate_unique_model_id(output_super_dir)
        for m, vals in metrics_results.items():
            save_summary(output_super_dir, cfg.method.name, cfg.task.name, cfg.task.num_simulations, model_id, m, vals, seed, cfg)
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
