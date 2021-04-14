import os
from sklearn.model_selection import ParameterGrid
from datetime import datetime
import pytz
import pandas as pd
import hashlib
import re

CWD = os.path.abspath('.')
TIMESTAMP = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d_%H-%M-%S')
GIT_COMMIT_HASH = os.popen('git rev-parse --short HEAD').read().replace('\n', '')
RESULTS_DIR = os.path.join(CWD, 'results', GIT_COMMIT_HASH)
if not os.path.exists(RESULTS_DIR):
    os.system('mkdir -p {}'.format(RESULTS_DIR))

PARAMS = {'act_func': ['linear'],
          'epoch_n': [100],
          'save_interval': [10],
          'r1_size': [100],
          's10': [10],
          's11': [1],
          's21': [10],
          's22': [1],
          's32': [5],
          'alpha_1': [0.1],
          'alpha_2': [5],
          'beta_1': [0.1],
          'beta_2': [0.1],
          'gamma_1': [0.01],
          'gamma_2': [0.01],
          'softmax_c': [20],
          'recog_mode': [1],
          'recog_value': [0],
          'in_dir': ['./data/CVCV_12'],
          'out_dir': [RESULTS_DIR],
          'timestamp': [TIMESTAMP],
          'ipynb': ['toy_model_kalman_variant_4.ipynb'],
          'weight_init_seed': [1],
          'timecourse_sharey': [True],
          'nlines_max': [12],
          'cohort_len': [2],
          'rhyme_len': [2],
          'plot_weights': [False]}

# save parameter grid
GRID = list(ParameterGrid(PARAMS))
GRID_DF = pd.DataFrame(GRID)

GRID_DF_KEEP = GRID_DF.drop(['epoch_n', 'save_interval', 'out_dir', 'timestamp', 'ipynb',
                             'timecourse_sharey', 'nlines_max', 'plot_weights'], axis=1)

GRID_DF['param_id'] = GRID_DF_KEEP.apply(lambda x: hashlib.sha1(x.to_json().encode()).hexdigest()[:10], axis=1)

for i in GRID_DF.index:
    PARAM_ID = GRID_DF.param_id[i]
    PKL_FILE = '{}_{}_{}.pkl'.format(TIMESTAMP, PARAM_ID, re.sub("\.ipynb$", "", GRID_DF.ipynb[i]))
    PKL_PATH = os.path.join(RESULTS_DIR, PKL_FILE)
    GRID_DF.loc[i].to_pickle(PKL_PATH)
    print(PKL_PATH)
