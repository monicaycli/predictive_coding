import pandas as pd
import glob
import os
import re

CWD = os.path.abspath(".")

GIT_COMMIT_HASH = os.popen("git rev-parse --short HEAD").read().replace("\n", "")
RESULTS_DIR = os.path.join(CWD, "results", GIT_COMMIT_HASH)

pkl_list = glob.glob(os.path.join(RESULTS_DIR, "**", "results.pkl"))

max_acc_list = {"id": [], "training": [], "max_acc": []}

for x in pkl_list:
    regex = re.compile(os.path.join(RESULTS_DIR, "(?P<id>.*)", "results.pkl"))
    results_id = regex.match(x).group("id")
    
    results_df = pd.read_pickle(x)

    results_df_g = results_df.groupby("training")
    
    for g in results_df_g.groups.keys():
        max_acc = results_df_g.get_group(g).groupby("epoch").mean()["accuracy"].max()
        
        max_acc_list["id"].append(results_id)
        max_acc_list["training"].append(g)
        max_acc_list["max_acc"].append(max_acc)
    
    max_acc_df = pd.DataFrame.from_dict(max_acc_list)
    max_acc_df.to_csv(os.path.join(RESULTS_DIR, "id_max_acc.csv"), index=False)