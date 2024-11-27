import json

from loaders.docalign12k import docalign12kLoader
def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'docalign12k':docalign12kLoader,
    }[name]
