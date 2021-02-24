import glob
import pandas as pd
import os
import sys

CWD = os.path.abspath('.')

if len(sys.argv) == 1:
    GIT_COMMIT_HASH = os.popen('git rev-parse --short HEAD').read().replace('\n', '')
else:
    GIT_COMMIT_HASH = sys.argv[1]

RESULTS_DIR = os.path.join(CWD, 'results', GIT_COMMIT_HASH)

results_list = glob.glob(os.path.join(RESULTS_DIR, "*_results.pkl"))

for i in results_list:
    filename = os.path.split(i)[1]
    df = pd.read_pickle(i)
    max_acc = df.groupby("training").get_group(False).groupby("epoch").mean().accuracy.max()
    print(filename, max_acc)
