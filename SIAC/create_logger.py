import os
import logging
file_path = os.path.dirname(os.path.realpath(__file__))                                                                                                                                                                 
version_file = file_path + '/VERSION'
with open(version_file, 'rb') as f:
    version = f.read().decode().strip()
def create_logger(fname = None):
    logger = logging.getLogger('SIAC-V%s'%version)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if fname is not None:
        fh = logging.FileHandler(fname)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)    
        logger.addHandler(fh)
    return logger


def create_component_progress_logger():
    component_progress_logger = logging.getLogger('ComponentProgress')
    component_progress_logger.setLevel(logging.INFO)
    component_progress_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    component_progress_logging_handler = logging.StreamHandler()
    component_progress_logging_handler.setLevel(logging.INFO)
    component_progress_logging_handler.setFormatter(component_progress_formatter)
    component_progress_logger.addHandler(component_progress_logging_handler)
    return component_progress_logger
