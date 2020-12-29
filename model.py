# -*- coding: utf-8 -*-
import numpy as np
import os

class Model:
    """
    This model has 3 levels.
    1: Several clusters of nodes which decode the input. Each cluster only takes part of the input.
    2: Intermediate level which integrates the entire input that's represented in the first level.
    3: The classifier.
    """
    def __init__(self, dataset, level1_module_size=32, level2_module_size=128):
        self.dtype = np.float32

        self.dataset = dataset
        self.iteration = 30
        
        # Activation function
        self.af = 'linear' # 'linear' or 'tanh'

        # Level 1 consists of multiple modules with 32 neurons in each moodule
        # Level-1 modules' receptive fields can be arranged into a 1D or 2D grid, with overlap between neighboring receptive fields
        self.level1_x = dataset.rf1_size[1]
        self.level1_y = dataset.rf1_size[0]
        self.level1_offset_x = dataset.rf1_offset_x
        self.level1_offset_y = dataset.rf1_offset_y
        self.level1_layout_x = dataset.rf1_layout_size[1]
        self.level1_layout_y = dataset.rf1_layout_size[0]
        self.level1_module_size = level1_module_size

        # Level 2 consists of one module with 128 neurons whose receptive field covers that of all level-1 modules
        self.level2_module_size = level2_module_size

        # Level 3 consists of localist nodes, one for each training image for classification
        self.level3_module_size = len(dataset.images)

        # Generative Weight Matrices
        # Between-level connections
        self.U1 = np.random.normal(loc=0, scale=0.1, size=(self.level1_layout_y, self.level1_layout_x, self.level1_y, self.level1_x, self.level1_module_size)).astype(self.dtype)
        self.U2 = np.random.normal(loc=0, scale=0.1, size=(self.level1_layout_y, self.level1_layout_x, self.level1_module_size, self.level2_module_size)).astype(self.dtype)
        self.U3 = np.random.normal(loc=0, scale=0.1, size=(self.level2_module_size, self.level3_module_size)).astype(self.dtype)

        # State Transmission Matrices
        # Within-level recurring connections
        self.V1 = np.random.normal(loc=0, scale=0.1, size=(self.level1_layout_y, self.level1_layout_x, self.level1_module_size, self.level1_module_size)).astype(self.dtype)
        self.V2 = np.random.normal(loc=0, scale=0.1, size=(self.level2_module_size, self.level2_module_size)).astype(self.dtype)
        self.V3 = np.random.normal(loc=0, scale=0.1, size=(self.level3_module_size, self.level3_module_size)).astype(self.dtype)

        # Learning Rates
        self.alpha_r = 1e-3
        self.alpha_u = 1e-3
        self.alpha_v = 1e-3

    def __f(self, x):
        """
        Applies activation function.
        """
        if self.af == 'linear':
            fx = x
        elif self.af == 'tanh':
            fx = np.tanh(x)
        return fx

    def __df(self, x):
        """
        Derivative of activation function.
        """
        if self.af == 'linear':
            dfx = np.ones(x.shape)
        elif self.af == 'tanh':
            dfx = (1 - np.square(np.tanh(x)))
        return dfx

    def __dW(self, r1, df_r10, e10, alpha):
        """
        Weight (U or V) update.

        Arguments:
        r1: current node activation value vector
        df_r10: predicted node activation
        e10: prediction error
        alpha: learning rate

        Returns: weight change to be applied
        """
        dW = np.outer(np.diag(df_r10) @ e10, r1)
        dW_alpha = alpha * dW
        return dW_alpha

    def __dr(self, U1, V1, df_r10, df_r11, e10, e11, e21, alpha, cross_entropy=False, training=False):
        """
        Node (r) update.

        To update a given level (e.g. r1), we consider the bottom-up and top-down prediction error.

        Arguments:
        U1: top-down weights from current level to lower level
        V1: recurrent weights of current level
        df_r10: derivative of bottom-up prediction
        df_r11: derivative of recurrent prediction
        e10: bottom-up prediction error
        e11: recurrent prediction error
        e21: top-down prediction error
        alpha: learning rate

        Returns: value change for node
        """
        if cross_entropy == True and training == False:
            dr1 = (U1.T @ np.diag(df_r10) @ e10) - ((1 - V1.T @ np.diag(df_r11)) @ e11)
        else:
            dr1 = (U1.T @ np.diag(df_r10) @ e10) - ((1 - V1.T @ np.diag(df_r11)) @ e11) - e21
        dr1_alpha = alpha * dr1
        return dr1_alpha

    def apply_input(self, inputs, labels, dataset, training):
        """
        Takes a 2D image as input. The 2D image is either a static input or time varying input, depending on how the
        image is preprocessed by `dataset.py`.

        Updates the models' between-level and within-level connections (instance vars).

        Also returns nodes and prediction error for reference.

        If the input is time varying, each time step can have >=1 iterations. In this case, each returned list will
        have elements corresponding to the value at a given iteration.

        If `training` is true:
        - The model weights are updated
        - The model uses `labels` as the target vector for the classifier level. If the input is time varying, `labels`
          will have as many vectors as there are time steps.

        inputs = rf2_patches is 4D: (rf2_layout_y, rf2_layout_x, rf2_y, rf2_x)
        labels is 3D: (rf2_layout_y, rf2_layout_x, number of images)
        """
        outputs = {'index':[],
                   'input': [],
                   'label': [],
                   'iteration': [],
                   'r1': [],
                   'r2': [],
                   'r3': [],
                   'e10': [],
                   'e21': [],
                   'e32': [],
                   'e43': [],
                   'e11': [],
                   'e22': [],
                   'e33': []}

        # inputs and estimates
        inputs = np.array(inputs, dtype=self.dtype)
        r1 = np.zeros((self.level1_layout_y, self.level1_layout_x, self.level1_module_size)).astype(self.dtype)
        r2 = np.zeros(self.level2_module_size).astype(self.dtype)
        r3 = np.zeros(self.level3_module_size).astype(self.dtype)
    
        for idx in np.ndindex(inputs.shape[:2]):
            I = dataset.get_rf1_patches_from_rf2_patch(inputs[idx])
            L = labels[idx]

            for i in range(self.iteration):
                # reshape
                I_x = I.reshape(I.shape[:2] + (-1,))
                U1_x = self.U1.reshape(I_x.shape + (r1.shape[-1],))

                # predictions
                ## between-level
                r10 = np.matmul(U1_x, np.expand_dims(r1, axis=-1)).reshape(I_x.shape)
                r21 = np.matmul(self.U2, np.expand_dims(r2, axis=-1)).reshape(r1.shape)
                r32 = self.U3 @ r3
                ## within-level
                r11 = np.matmul(self.V1, np.expand_dims(r1, axis=-1)).reshape(r1.shape)
                r22 = self.V2 @ r2
                r33 = self.V3 @ r3

                # apply activation function on predictions
                ## between-level
                f_r10 = self.__f(r10)
                f_r21 = self.__f(r21)
                f_r32 = self.__f(r32)
                ## within-level
                f_r11 = self.__f(r11)
                f_r22 = self.__f(r22)
                f_r33 = self.__f(r33)

                # apply derivative of activation function on predictions
                ## between-level
                df_r10 = self.__df(r10)
                df_r21 = self.__df(r21)
                df_r32 = self.__df(r32)
                ## within-level
                df_r11 = self.__df(r11)
                df_r22 = self.__df(r22)
                df_r33 = self.__df(r33)

                # prediction errors
                ## between-level
                e10 = I_x - f_r10
                e21 = r1 - f_r21
                e32 = r2 - f_r32
                e43 = (np.exp(r3)/np.sum(np.exp(r3))) - L
                ## within-level
                e11 = r1 - f_r11
                e22 = r2 - f_r22
                e33 = r3 - f_r33

                # calculate r updates
                dr1 = np.array([self.__dr(U1_x[j,k], self.V1[j,k], df_r10[j,k], df_r11[j,k], e10[j,k], e11[j,k], e21[j,k], alpha = self.alpha_r, \
                                          cross_entropy = False, training = training) for j,k in np.ndindex(I.shape[:2])]).reshape(r1.shape)
                dr2 = sum([self.__dr(self.U2[j,k], self.V2, df_r21[j,k], df_r22, e21[j,k], e22, e32, alpha = self.alpha_r, \
                                     cross_entropy = False, training = training) for j,k in np.ndindex(I.shape[:2])])
                dr3 = self.__dr(self.U3, self.V3, df_r32, df_r33, e32, e33, e43, alpha = self.alpha_r, cross_entropy = True, training = training)
                
                # calculate U and V updates
                if training == True:
                    dU1 = np.array([self.__dW(r1[j,k], df_r10[j,k], e10[j,k], alpha = self.alpha_u) for j,k in np.ndindex(I.shape[:2])]).reshape(self.U1.shape)
                    dU2 = np.array([self.__dW(r2, df_r21[j,k], e21[j,k], alpha = self.alpha_u) for j,k in np.ndindex(I.shape[:2])]).reshape(self.U2.shape)
                    dU3 = self.__dW(r3, df_r32, e32, alpha = self.alpha_u)

                    dV1 = np.array([self.__dW(r1[j,k], df_r11[j,k], e11[j,k], alpha = self.alpha_v) for j,k in np.ndindex(I.shape[:2])]).reshape(self.V1.shape)
                    dV2 = self.__dW(r2, df_r22, e22, alpha = self.alpha_v)
                    dV3 = self.__dW(r3, df_r33, e33, alpha = self.alpha_v)

                # apply r updates
                r1 += dr1
                r2 += dr2
                r3 += dr3

                # apply U and V updates
                if training == True:
                    self.U1 += dU1
                    self.U2 += dU2
                    self.U3 += dU3
                    self.V1 += dV1
                    self.V2 += dV2
                    self.V3 += dV3
                
                # append outputs
                outputs['index'].append(idx)
                outputs['input'].append(inputs[idx])
                outputs['label'].append(labels[idx])
                outputs['iteration'].append(i)
                outputs['r1'].append(r1.copy())
                outputs['r2'].append(r2.copy())
                outputs['r3'].append(r3.copy())
                outputs['e10'].append(e10.copy())
                outputs['e21'].append(e21.copy())
                outputs['e32'].append(e32.copy())
                outputs['e43'].append(e43.copy())
                outputs['e11'].append(e11.copy())
                outputs['e22'].append(e22.copy())
                outputs['e33'].append(e33.copy())

        return outputs

    def train(self, dataset):
        """
        Training on rf2_patches in order within a given image.
        Images are presented in the order defined in dataset.

        rf2 patches of a given image are presented in an ascending sequence (moves through x before y)
        inputs = rf2_patches is 4D: (rf2_layout_y, rf2_layout_x, rf2_y, rf2_x)
        labels is 3D: (rf2_layout_y, rf2_layout_x, number of images)
        """
        for i in range(dataset.rf2_patches.shape[0]):
            inputs = dataset.rf2_patches[i]
            labels = dataset.labels[i]
            outputs = self.apply_input(inputs, labels, dataset, training=True)

        print("train finished")

    def reconstruct(self, r, level=1):
        """
        Reconstructs what the model thinks the image looks like, from any level of node activation.
        """
        # Representations at each level. Values change with input.
        if level == 1:
            r1 = r
        elif level == 2:
            r2 = r
            r1 = self.U2.dot(r2).reshape((self.level1_layout_y, self.level1_layout_x, self.level1_module_size))
        elif level == 3:
            r3 = r
            r2 = self.U3.dot(r3)
            r1 = self.U2.dot(r2).reshape((self.level1_layout_y, self.level1_layout_x, self.level1_module_size))
            
        rf2_patch = np.zeros((self.level1_y + (self.level1_offset_y * (self.level1_layout_y - 1)), \
                              self.level1_x + (self.level1_offset_x * (self.level1_layout_x - 1))), dtype=self.dtype)
        
        # reconstruct each of the three rf1 patches separately and then combine
        for i, j in np.ndindex(self.level1_layout_y, self.level1_layout_x):
            r = r1[i, j]
            U = self.U1[i, j]
            Ur = U.dot(r)
            rf2_patch[self.level1_offset_y * i : self.level1_offset_y * i + self.level1_y, \
                      self.level1_offset_x * j : self.level1_offset_x * j + self.level1_x] += Ur
        return rf2_patch

    def get_level2_rf(self, index):
        """
        rf: receptive field
        values don't change with input (when model is not being trained)
        """
        rf = np.zeros((self.level1_y + (self.level1_offset_y * (self.level1_layout_y - 1)), \
                       self.level1_x + (self.level1_offset_x * (self.level1_layout_x - 1))), dtype=self.dtype)

        for i, j in np.ndindex(self.level1_layout_y, self.level1_layout_x):
            U2 = self.U2[i, j, :, index]
            UU = self.U1[i, j].dot(U2)
            rf[self.level1_offset_y * i :self.level1_offset_y * i + self.level1_y, \
               self.level1_offset_x * j :self.level1_offset_x * j + self.level1_x] += UU

        return rf

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
                            U3=self.U3,
                            V1=self.V1,
                            V2=self.V2,
                            V3=self.V3)
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
        self.U3 = data["U3"]
        self.V1 = data["V1"]
        self.V2 = data["V2"]
        self.V3 = data["V3"]
        print("loaded: {}".format(dir_name))
