import yaml
import os
from pathlib import Path

DIRNAME = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()


def get_descriptors():
    '''Get description'''
    with open(os.path.join(DIRNAME, "settings", "descriptor.yaml"), "r") as handle:
        descriptor = yaml.safe_load(handle.read())
    return descriptor
