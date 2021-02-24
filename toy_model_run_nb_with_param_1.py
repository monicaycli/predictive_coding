import sys
import os
import pandas as pd
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

CWD = os.path.abspath('.')

PKL_PATH = sys.argv[1]
OUT_PATH = os.path.splitext(PKL_PATH)[0] + '.ipynb'

param = pd.read_pickle(PKL_PATH)
print(param)

nb = nbformat.read(param.ipynb, as_version=4)
nb['cells'].insert(2, nbformat.v4.new_code_cell('param = {}'.format(str(param.to_dict()))))
ep = ExecutePreprocessor(timeout=None)
ep.preprocess(nb, {'metadata': {'path': CWD}})

nbformat.write(nb, OUT_PATH)
os.system('jupyter nbconvert {} --to pdf'.format(OUT_PATH))
os.system('rm {}'.format(OUT_PATH))
