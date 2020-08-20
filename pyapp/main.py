import logging
import argparse

from main_module import MainModule

def init_logger():
    '''
    Initiate main logger.
    '''
    _logger = logging.getLogger('Main')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)
    return _logger

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True, \
        help='Device number of camera')
    parser.add_argument('--padding-size', type=int, required=True, \
        help='Padding size for cropping image')
    args = parser.parse_args()

    # Logging
    logger = init_logger()
    logger.info('Device Number : %d', args.device)
    logger.info('Padding size : %d', args.padding_size)

    # main(logger, args)
    mm = MainModule(args.device, args.padding_size)
    mm.run()
