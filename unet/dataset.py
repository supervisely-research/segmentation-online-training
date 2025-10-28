import json

import utils


N_CLASSES = 10


with utils.DATA_ROOT.joinpath('config.json').open('rt') as f:
    CONFIG = json.load(f)


CLS_RATIOS = None