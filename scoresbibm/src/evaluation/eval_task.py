

from scoresbibm.src.tasks.base_task import InferenceTask, AllConditionalTask

import jax
import time



def eval_inference_task(task: InferenceTask, model, metric_fn, metric_params, rng, **kwargs):
    metric_values = []
    metric_params = dict(metric_params)
    condition_mask_fn = metric_params.pop("condition_mask_fn", "structured_random")
    if condition_mask_fn != "posterior":
        return None, None
    average_sampling_time = 0
    for i in task.observations:
        rng_metric, rng_metric_i = jax.random.split(rng)
        x_o = task.get_observation(i)
        true_posterior_samples = task.get_reference_posterior_samples(i)
        start_time = time.time()
        est_posterior_samples = model.sample(num_samples=true_posterior_samples.shape[0], x_o=x_o, rng=rng_metric_i)
        sampling_time = time.time() - start_time
        val = metric_fn(true_posterior_samples, est_posterior_samples, rng=rng_metric, **metric_params)
        print("Metric value: ", val)
        metric_values.append(val)
        average_sampling_time += sampling_time / len(task.observations)
    return metric_values, average_sampling_time


def eval_all_conditional_task(task: AllConditionalTask, model, metric_fn, metric_params, rng, num_samples=2000, num_evaluations=2):
    metric_values = []
    average_sampling_time = 0
    metric_params = dict(metric_params)
    condition_mask_fn = metric_params.pop("condition_mask_fn", "structured_random")
    reference_sampler = task.get_reference_sampler()
    observation_generator = task.get_observation_generator(condition_mask_fn=condition_mask_fn)
    
    rng, rng_obs = jax.random.split(rng)
    observation_stream = observation_generator(rng_obs)
    for i in range(num_evaluations):
        rng, rng_metric, rng_sample_ref, rng_sample_model = jax.random.split(rng,4)
        condition_mask, x_o, theta_o = next(observation_stream)
        print("Conditional: ", condition_mask,x_o)
        start_time = time.time()
        est_posterior_samples = model.sample(num_samples, x_o= x_o,condition_mask=condition_mask, rng=rng_sample_model)
        sampling_time = time.time() - start_time
        true_posterior_samples = reference_sampler.sample(num_samples, x_o= x_o,condition_mask=condition_mask,rng=rng_sample_ref)  
        metric_value = metric_fn(est_posterior_samples, true_posterior_samples, rng=rng_metric, **metric_params)
        print("Metric value: ", metric_value)
        metric_values.append(metric_value)
        average_sampling_time += sampling_time / num_evaluations
    return metric_values, average_sampling_time

def eval_unstructured_task(task, model, metric_fn, metric_params, rng, num_samples=1000, num_evaluations=50):
    metric_values = []
    average_sampling_time = 0
    metric_params = dict(metric_params)
    condition_mask_fn = metric_params.pop("condition_mask_fn", "structured_random")
    reference_sampler = task.get_reference_sampler()
    observation_generator = task.get_observation_generator(condition_mask_fn=condition_mask_fn)
    
    rng, rng_obs = jax.random.split(rng)
    observation_stream = observation_generator(rng_obs)
    for i in range(num_evaluations):
        rng, rng_metric, rng_sample_ref, rng_sample_model = jax.random.split(rng,4)
        condition_mask, x_o, theta_o, meta_data, node_id  = next(observation_stream)
        print("Conditional: ", condition_mask,x_o)
        start_time = time.time()
        est_posterior_samples = model.sample(num_samples, x_o= x_o,condition_mask=condition_mask,node_id=node_id, meta_data=meta_data, rng=rng_sample_model)
        sampling_time = time.time() - start_time
        true_posterior_samples = reference_sampler.sample(num_samples, x_o= x_o,condition_mask=condition_mask,meta_data=meta_data,rng=rng_sample_ref)  
        metric_value = metric_fn(est_posterior_samples, true_posterior_samples, rng=rng_metric, **metric_params)
        print("Metric value: ", metric_value)
        metric_values.append(metric_value)
        average_sampling_time += sampling_time / num_evaluations
    return metric_values, average_sampling_time