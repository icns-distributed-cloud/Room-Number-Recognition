import json
import logging
import argparse

from main_engine import MainEngine

def init_logger():
    '''
    Initiate main logger.
    '''
    _logger = logging.getLogger('Main')
    _logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)
    return _logger

def load_config(filename):
    '''
    Load and parse config file.
    @params filename: config file path
    '''
    cfg: dict
    with open(filename, 'r') as f:
        cfg = json.load(f)
    return cfg

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, \
        default='config.json', help='config file')
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Logging
    logger = init_logger()

    # main(logger, args)
    me = MainEngine(cfg)
    me.run()