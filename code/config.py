
import json




def load_config(file_path):
    
    with open(file_path) as f:
        config = json.load(f)
    
    return config

def config_is_valid(config):

    required_keys = ['color_space', 'orientations', 
                     'pix_per_cell', 'cell_per_block', 
                     'spatial_size', 'histogram_bins']


    if len(required_keys) != len(config):
        return False


    if list(config.keys()) != required_keys:
        return False

    return True
