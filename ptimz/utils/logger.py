import logging
import os


def get_logger(dir_path=None, name='dlvision_logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if dir_path is not None:
        fh = logging.FileHandler(os.path.join(dir_path, f'{name}.log'))
        fh.setLevel(logging.DEBUG)
    else:
        fh = None
    # stream handler will send message to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if fh is not None:
        fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    use_tqdm = os.getenv('use_tqdm')
    use_tqdm = True if use_tqdm and '1' == use_tqdm else False
    if use_tqdm:
        if fh is not None:
            logger.addHandler(fh)
    else:
        if fh is not None:
            logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
