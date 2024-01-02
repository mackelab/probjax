

from scoresbibm.src.tasks.base_task import InferenceTask, AllConditionalTask

import jax
import time



def eval_inference_task(task: InferenceTask, model, metric_fn, metric_params, rng):
    metric_values = []
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


def eval_all_conditional_task(task: AllConditionalTask, model, metric_fn, metric_params, rng, num_samples=2000, num_evaluations=100):
    metric_values = []
    average_sampling_time = 0
    reference_sampler = task.get_reference_sampler()
    observation_generator = task.get_observation_generator()
    
    rng, rng_obs = jax.random.split(rng)
    observation_stream = observation_generator(rng_obs)
    for i in range(num_evaluations):
        rng, rng_metric, rng_sample_ref, rng_sample_model = jax.random.split(rng,4)
        condition_mask, x_o, theta_o = next(observation_stream)
        true_posterior_samples = reference_sampler.sample(num_samples, condition_mask=condition_mask, x_o = x_o, rng=rng_sample_ref)
        start_time = time.time()
        est_posterior_samples = model.sample(num_samples=true_posterior_samples.shape[0], condition_mask=condition_mask, x_o=x_o, rng=rng_sample_model)
        sampling_time = time.time() - start_time
        metric_value = metric_fn(true_posterior_samples, est_posterior_samples, rng=rng_metric, **metric_params)
        print("Conditional: ", condition_mask)
        print("Metric value: ", metric_value)
        metric_values.append(metric_value)
        average_sampling_time += sampling_time / num_evaluations
    return metric_values, average_sampling_time