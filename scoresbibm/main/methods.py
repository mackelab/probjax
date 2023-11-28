
import sbi 
from sbi.utils import posterior_nn, likelihood_nn, classifier_nn
from sbi.inference import SNPE, SNLE, SNRE

from main.score_based_sbi import train_conditional_score_model


class ModelWrapper:
    def __init__(self, model, method):
        self.model = model
        self.method = method
        
    def sample(self, num_samples, x_o = None, rng=None, **kwargs):
        if self.method in ["npe", "nle", "nre"]:
            return self.model.sample((num_samples,), x=x_o)
        else:
            raise NotImplementedError()
        
    def log_prob(self, theta, x_o, **kwargs):
        if self.method == "npe":
            return self.model.log_prob(theta, x=x_o)
        else:
            raise NotImplementedError()
        


def run_npe_default(task,thetas, xs, method_cfg, rng=None):
    """ Train a default SBI model"""
    device = method_cfg.device
    density_estimator = posterior_nn(**method_cfg.params_posterior_nn)
    inference = SNPE(density_estimator=density_estimator, device=device)
    _ = inference.append_simulations(thetas, xs)
    
    # Train
    density_estimator = inference.train(**method_cfg.params_train)
    
    # Output is sampling_fn
    posterior = inference.build_posterior()
    
    model = ModelWrapper(posterior, method="npe")
    return model


def run_nle_default(task, thetas, xs, method_cfg, rng=None):
    device = method_cfg.device
    density_estimator = likelihood_nn(**method_cfg.params_likelihood_nn)
    inference = SNLE(prior = task.get_prior(),density_estimator=density_estimator, device=device)
    _ = inference.append_simulations(thetas, xs)
    
    # Train
    density_estimator = inference.train(**method_cfg.params_train)
    
    posterior = inference.build_posterior(**method_cfg.params_build_posterior)
    model = ModelWrapper(posterior, method="nle")
    return model


def run_nre_default(task, thetas, xs, method_cfg, rng=None):
    device = method_cfg.device
    classifier = classifier_nn(**method_cfg.params_classifier_nn)
    inference = SNRE(prior = task.get_prior(), classifier=classifier, device=device)
    _ = inference.append_simulations(thetas, xs)
    
    # Train
    density_estimator = inference.train(**method_cfg.params_train)
    
    posterior = inference.build_posterior(**method_cfg.params_build_posterior)
    model = ModelWrapper(posterior, method="nre")
    return model




def run_nspe(task, thetas, xs, method_cfg, rng=None):
    model = train_conditional_score_model(task, thetas, xs, method_cfg, rng)
    return model
    



def get_method(name:str):
    """ Get a method"""
    if name == "npe":
        return run_npe_default
    elif name == "nle":
        return run_nle_default
    elif name == "nre":
        return run_nre_default
    elif name == "nspe":
        return run_nspe
    else:
        raise NotImplementedError()