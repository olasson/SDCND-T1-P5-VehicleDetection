
import argparse

from code.config import load_config, config_is_valid
from code.misc import file_exists

ERROR_PREFIX = 'ERROR_MAIN: '


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Advanced Lane Lines')

    parser.add_argument(
        '--video',
        type = str,
        nargs = '?',
        default = '',
        help = 'Path to a folder video file to run the pipeline on.',
    )


    parser.add_argument(
        '--config',
        type = str,
        nargs = '?',
        default = './config/config1.json',
        help = 'Path to a .json file containing project config.',
    )

    args = parser.parse_args()

    # Config setup

    file_path_config = args.config

    if not file_exists(file_path_config):
        print(ERROR_PREFIX + 'The config file: ' + file_path_config + ' does not exist!')
        exit()

    config = load_config(file_path_config)

    if not config_is_valid(config):
        print(ERROR_PREFIX + 'Incorrectly structured config file at: ' + file_path_config + '! Check code/config.py for more info!')
        exit()

