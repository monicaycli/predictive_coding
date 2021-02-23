import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby

def softmax(x, c=1):
    x = np.array(x)
    x = x - x.max()
    return np.exp(x*c)/np.sum(np.exp(x*c))

def f(x, af="linear"):
    '''
    Applies activation function.
    '''
    if af == "linear":
        fx = x
    elif af == "tanh":
        fx = np.tanh(x)
    return fx

def f_prime(x, af="linear"):
    '''
    Derivative of activation function.
    '''
    if af == "linear":
        dfx = np.eye(x.size)
    elif af == "tanh":
        dfx = np.diag(1 - np.square(np.tanh(x)))
    return dfx

def raw_vs_softmax_plots(df, ncols, nrows, subplot_yx, scale, groupby, value, c=1, sharex=False, sharey=False, cmap=None):
    subplot_y, subplot_x = tuple(i * scale for i in (subplot_yx))

    fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_x*ncols, subplot_y*nrows), sharex=sharex, sharey=sharey)
    
    df_groupby = df.groupby(groupby)
    group_keys = df_groupby.groups.keys()
    
    for k in group_keys:
        k_idx = list(group_keys).index(k)

        raw_r = df_groupby.get_group(k).apply(lambda x: pd.Series(x[value]), axis=1).reset_index(drop=True)
        softmaxd_r = df_groupby.get_group(k).apply(lambda x: pd.Series(softmax(x[value], c=c)), axis=1).reset_index(drop=True)

        raw_r.columns = group_keys
        softmaxd_r.columns = group_keys

        raw_plot = raw_r.plot(ax=axes[k_idx, 0], title="{}: raw activations".format(k), cmap=cmap);
        softmaxd_plot = softmaxd_r.plot(ax=axes[k_idx, 1], title="{}: softmax'd activations".format(k), cmap=cmap);
        
        raw_plot.legend(loc="lower left");
        softmaxd_plot.legend(loc="lower left");
        
        # thicken target line
        for i, l1, l2 in zip(range(len(group_keys)), raw_plot.lines, softmaxd_plot.lines):
            if i == k_idx:
                plt.setp(l1, linewidth=3)
                plt.setp(l2, linewidth=3)

    plt.tight_layout();
    
def recon_plots(df, ncols, nrows, subplot_yx, scale, groupby, value, sharex=True, sharey=True, cmap="gray", y_label_list=None):
    subplot_y, subplot_x = tuple(i * scale for i in (subplot_yx))

    fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_x*ncols, subplot_y*nrows), sharex=sharex, sharey=sharey)

    df_groupby = df.groupby(groupby)
    group_keys = df_groupby.groups.keys()
    
    vmin = df.apply(lambda x: pd.Series(x[value]), axis=1).values.min()
    vmax = df.apply(lambda x: pd.Series(x[value]), axis=1).values.max()
    
    for k, ax in zip(group_keys, axes.flatten()):
        vall = df_groupby.get_group(k).apply(lambda x: pd.Series(x[value]), axis=1).values.T
        im = ax.imshow(vall, cmap=cmap, vmin=vmin, vmax=vmax)
    
        if y_label_list != None:
            ax.set_yticks(range(len(y_label_list)))
            ax.set_yticklabels(y_label_list)

        ax.set_title(k)

    fig.colorbar(im, ax=axes.ravel().tolist());
    
def argmax_1(x):
    '''
    refined argmax which only returns the index when there is only one element with the maximum value, otherwise returns NaN
    '''
    x = np.array(x)
    max_marks = [1 if i == x.max() else 0 for i in x]
    argmax_x = np.argmax(max_marks) if np.sum(max_marks) == 1 else np.nan
    return argmax_x

def recog(L, mode=1, value=1):
    '''
    Operationalize word recognition. Input L should be a vector of raw or transformed activation values.
    
    Modes:
    1. Most activated one at the last one time step (or a specific number of time steps).
    2. Select the node that first reaches and stays at the maximum for a certain number of time steps
    3. Absolute threshold at specified value.
    '''
    if mode == 1:
        value = 1
        L_argmax = L.apply(lambda x: argmax_1(x))
        recog_node = L_argmax.tail(value)
    elif mode == 2:
        L_argmax = L.apply(lambda x: argmax_1(x))
        X = np.array([[k, sum(1 for i in g)] for k, g in groupby(L_argmax)])
        df = pd.DataFrame(X, columns=["max_node", "steps"])
        max_nodes = df.query("steps >= {}".format(value))["max_node"]
        recog_node = max_nodes.head(1) if len(max_nodes) > 0 else np.nan
    elif mode == 3:
        L_argmax = L.apply(lambda x: argmax_1(x > value))
        X = np.array([[k, sum(1 for i in g)] for k, g in groupby(L_argmax)])
        df = pd.DataFrame(X, columns=["max_node", "steps"])
        max_nodes = df.query("max_node.notna()", engine="python")["max_node"]
        recog_node = max_nodes.head(1) if len(max_nodes) > 0 else np.nan

    return recog_node