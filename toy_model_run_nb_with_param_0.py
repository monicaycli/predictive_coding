import os
from sklearn.model_selection import ParameterGrid
from datetime import datetime
import pytz
import pandas as pd
import hashlib

CWD = os.path.abspath('.')

GIT_COMMIT_HASH = os.popen('git rev-parse --short HEAD').read().replace('\n', '')
RESULTS_DIR = os.path.join(CWD, 'results', GIT_COMMIT_HASH)
if not os.path.exists(RESULTS_DIR):
    os.system('mkdir -p {}'.format(RESULTS_DIR))

PARAMS = {'act_func': ['linear', 'tanh'],
          'epoch_n': [1000],
          'save_interval': [50],
          'r1_size': [100],
          's10': [10],
          's11': [1],
          's21': [10],
          's22': [1],
          's32': [5],
          'alpha_1': [0.1],
          'alpha_2': [10],
          'beta_1': [0.1],
          'beta_2': [0.1],
          'beta_3': [0.1],
          'beta_f': [0.1],
          'gamma_1': [0.01],
          'gamma_2': [0.01],
          'softmax_c': [1],
          'recog_mode': [1],
          'recog_value': [1],
          'in_dir': ['./data/3x3']}

# save parameter grid
GRID = list(ParameterGrid(PARAMS))
GRID_DF = pd.DataFrame(GRID)

TIMESTAMP = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d_%H-%M-%S')
GRID_DF['timestamp'] = TIMESTAMP

GRID_DF_KEEP = GRID_DF.drop(['epoch_n', 'save_interval', 'timestamp'], axis=1)

# save simulations with the same set of parameters to the same directory (with hashing)
GRID_DF['out_dir'] = GRID_DF_KEEP.apply(lambda x: 
                                        os.path.join(RESULTS_DIR, hashlib.sha1(x.to_json().encode()).hexdigest()[:10]),
                                        axis=1)

for i in GRID_DF.index:
    OUT_DIR_SPLIT = os.path.split(GRID_DF.out_dir[i])
    PKL_PATH = os.path.join(OUT_DIR_SPLIT[0], '{}_{:03d}_{}.pkl'.format(TIMESTAMP, i, OUT_DIR_SPLIT[1]))
    GRID_DF.loc[i].to_pickle(PKL_PATH)
    print(PKL_PATH)
