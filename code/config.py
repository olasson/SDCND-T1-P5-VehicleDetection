
import json

def load_config(file_path):
    
    with open(file_path) as f:
        config = json.load(f)
    
    return config

def config_is_valid(config, template = './config/config_template.json'):

    config_template = load_config(template)

    if len(config_template) != len(config):
        return False


    if list(config.keys()) != list(config_template.keys()):
        return False

    return True
