import os

from Incept.utils import load_config

def load_exp_config(method, dataset):
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(config_dir, method, f"{method}_{dataset}.yaml")
    return load_config(config_file)