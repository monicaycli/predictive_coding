import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
import os

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

def raw_vs_softmax_plots(df, nrows, subplot_yx, scale, groupby, value, nodes, nlines_max=10, c=1, sharex=False, sharey=False, cmap=None):
    df_groupby = df.groupby(groupby)
    group_keys = list(df_groupby.groups.keys())
    
    ncols = 2
    nrows = nrows if nrows <= len(group_keys) else len(group_keys)
    
    subplot_y, subplot_x = tuple(i * scale for i in (subplot_yx))

    fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_x*ncols, subplot_y*nrows), sharex=sharex, sharey=sharey)
    
    for k in group_keys[:nrows]:
        k_idx = group_keys.index(k)

        raw_r = df_groupby.get_group(k).apply(lambda x: pd.Series(x[value]), axis=1).reset_index(drop=True)
        softmaxd_r = df_groupby.get_group(k).apply(lambda x: pd.Series(softmax(x[value], c=c)), axis=1).reset_index(drop=True)

        raw_r.columns = nodes
        softmaxd_r.columns = nodes
        
        if len(nodes) > nlines_max:
            raw_r = raw_r.filter(items=raw_r.tail(1).squeeze().sort_values(ascending=False).head(nlines_max).index)
            softmaxd_r = softmaxd_r.filter(items=softmaxd_r.tail(1).squeeze().sort_values(ascending=False).head(nlines_max).index)

        raw_plot = raw_r.plot(ax=axes[k_idx, 0], title="{}: raw activations".format(k), cmap=cmap);
        softmaxd_plot = softmaxd_r.plot(ax=axes[k_idx, 1], title="{}: softmax'd activations".format(k), cmap=cmap);
        
        raw_plot.legend(loc="lower left");
        softmaxd_plot.legend(loc="lower left");
        
        # thicken target line
        for i, j, l1, l2 in zip(raw_r.columns, softmaxd_r.columns, raw_plot.lines, softmaxd_plot.lines):
            if i == k:
                plt.setp(l1, linewidth=3)
            if j == k:
                plt.setp(l2, linewidth=3)

    plt.tight_layout();
    
def recon_plots(df, ncols, nrows, subplot_yx, scale, groupby, value, vmin=None, vmax=None, sharex=True, sharey=True, cmap="gray", y_label_list=None):
    subplot_y, subplot_x = tuple(i * scale for i in (subplot_yx))

    fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_x*ncols, subplot_y*nrows), sharex=sharex, sharey=sharey)

    df_groupby = df.groupby(groupby)
    group_keys = df_groupby.groups.keys()
    
    if vmax == None:
        vmax = df.apply(lambda x: np.abs(pd.Series(x[value])), axis=1).values.max()
    if vmin == None:
        vmin = -vmax
    
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

def category_given_target(target, word):

    if target == word:
        category = 'target'
    elif target[0:1] == word[:1]:
        category = 'cohort'
    elif target[1:] == word[1:]:
        category = 'rhyme'
    elif word in target:
        category = 'embedded'
    else:
        category = 'other'

    return category

class Model:
    def __init__(self, I_size, r1_size, r2_size, L_size, seed=None):
        self.I_size = I_size
        self.r1_size = r1_size
        self.r2_size = r2_size
        self.L_size = L_size
        
        np.random.seed(seed)
        self.U1 = np.random.normal(loc=0, scale=0.1, size=(I_size, r1_size))
        self.U2 = np.random.normal(loc=0, scale=0.1, size=(r1_size, r2_size))
        self.V1 = np.random.normal(loc=0, scale=0.1, size=(r1_size, r1_size))
        self.V2 = np.random.normal(loc=0, scale=0.1, size=(r2_size, r2_size))
        
        self.s10 = 1
        self.s11 = 1
        self.s21 = 1
        self.s22 = 1
        self.s32 = 1
        
        self.alpha_1 = 1
        self.alpha_2 = 1
        self.beta_1 = 1
        self.beta_2 = 1
        self.gamma_1 = 1
        self.gamma_2 = 1
        
    def apply_input(self, label, I, L, training=False, af="linear"):
        output = {"training": [], "label": [], "I": [], "L": [], "timestep": [],
                  "r1_bar": [], "r2_bar": [], "r2_bar_x": [],
                  "r1_hat": [], "r2_hat": [], "r2_hat_x": []}
        
        s10 = self.s10
        s11 = self.s11
        s21 = self.s21
        s22 = self.s22
        s32 = self.s32
        
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        gamma_1 = self.gamma_1
        gamma_2 = self.gamma_2
        
        r1_hat = np.zeros(self.r1_size)
        r2_hat = np.zeros(self.r2_size)
        r2_hat_x = np.zeros(self.r2_size)
        
        U1_hat = self.U1.copy()
        U2_hat = self.U2.copy()
        
        V1_hat = self.V1.copy()
        V2_hat = self.V2.copy()

        for idx in np.arange(I.shape[1]):
            r1_hat_old = r1_hat.copy()
            r2_hat_old = r2_hat.copy()
            r2_hat_x_old = r2_hat_x.copy()

            U1_bar = U1_hat.copy()
            U2_bar = U2_hat.copy()

            V1_bar = V1_hat.copy()
            V2_bar = V2_hat.copy()
            
            r1_bar = f(V1_bar @ r1_hat_old, af=af)
            r2_bar = f(V2_bar @ r2_hat_old, af=af)
            r2_bar_x = f(V2_bar @ r2_hat_x_old, af=af)

            r1_hat = r1_bar + alpha_1/s10 * U1_bar.T @ f_prime(U1_bar @ r1_bar, af=af) @ (I[:, idx] - f(U1_bar @ r1_bar, af=af)) - alpha_1/s21 * (r1_bar - f(U2_bar @ r2_bar, af=af))
            r2_hat = r2_bar + alpha_2/s21 * U2_bar.T @ f_prime(U2_bar @ r2_bar, af=af) @ (r1_bar - f(U2_bar @ r2_bar, af=af))
            r2_hat_x = r2_hat - 1/2 * alpha_2/s32 * (softmax(r2_bar) - L)

            if training == True:
                U1_hat = U1_bar + beta_1/s10 * f_prime(U1_bar @ r1_hat, af=af) @ np.outer(I[:, idx] - f(U1_bar @ r1_hat, af=af), r1_hat)
                U2_hat = U2_bar + beta_2/s21 * f_prime(U2_bar @ r2_hat_x, af=af) @ np.outer(r1_bar - f(U2_bar @ r2_hat_x, af=af), r2_hat_x)

                V1_hat = V1_bar + gamma_1/s11 * f_prime(V1_bar @ r1_hat_old, af=af) @ np.outer(r1_hat - f(V1_bar @ r1_hat_old, af=af), r1_hat_old)
                V2_hat = V2_bar + gamma_2/s22 * f_prime(V2_bar @ r2_hat_x_old, af=af) @ np.outer(r2_hat_x - f(V2_bar @ r2_hat_x_old, af=af), r2_hat_x_old)
            
            output["training"].append(training)
            output["label"].append(label)
            output["I"].append(I[:, idx])
            output["L"].append(L)
            output["timestep"].append(idx)

            output["r1_bar"].append(r1_bar.copy())
            output["r2_bar"].append(r2_bar.copy())
            output["r2_bar_x"].append(r2_bar_x.copy())
            
            output["r1_hat"].append(r1_hat.copy())
            output["r2_hat"].append(r2_hat.copy())
            output["r2_hat_x"].append(r2_hat_x.copy())
            
        self.U1 = U1_hat.copy()
        self.U2 = U2_hat.copy()
        
        self.V1 = V1_hat.copy()
        self.V2 = V2_hat.copy()
        
        output_df = pd.DataFrame.from_dict(output)
        return output_df
    
    def save(self, dir_name):
        """
        Saves all weights to disk.
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_path = os.path.join(dir_name, "model") 

        np.savez_compressed(file_path,
                            U1=self.U1,
                            U2=self.U2,
                            V1=self.V1,
                            V2=self.V2)
        print("saved: {}".format(dir_name))

    def load(self, dir_name):
        """
        Load previously saved weights from disk.
        """
        file_path = os.path.join(dir_name, "model.npz")
        if not os.path.exists(file_path):
            print("saved file not found")
            return
        
        data = np.load(file_path)
        self.U1 = data["U1"]
        self.U2 = data["U2"]
        self.V1 = data["V1"]
        self.V2 = data["V2"]
        print("loaded: {}".format(dir_name))
