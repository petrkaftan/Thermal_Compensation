import pandas as pd
import numpy as np
import gp_model as mod
import matplotlib.pyplot as plt

class Thermal_Compensation:

    def __init__(self):
        pass

    def set_train_start(self,train_start):
        self.train_start = train_start

    def set_train_end(self,train_end):
        self.train_end  = train_end
        self.pred_start = self.train_end+1

    def set_e_threshold(self,e_threshold):
        self.e_threshold = e_threshold

    def set_std_threshold(self,std_threshold):
        self.std_threshold = std_threshold

    def set_N_skip(self,N_skip):
        self.N_skip = N_skip

    def append_data(self,df_X: pd.DataFrame, df_y: pd.DataFrame):

        self.df_X = df_X
        self.df_y = df_y

        self.pred_end = self.df_X.shape[0]

        self.df_X_train = self.df_X.iloc[self.train_start:self.train_end,:]
        self.df_X_pred  = self.df_X.iloc[self.pred_start:self.pred_end,:]

        self.df_y_train = self.df_y.iloc[self.train_start:self.train_end]
        self.df_y_val   = self.df_y.iloc[self.pred_start:self.pred_end]

    def train_model(self):

        self.model = mod.GPy_module()
        self.model.train_model(self.df_X_train,self.df_y_train)

    def predict(self,recalibrations=True):

        if recalibrations == False:

            self.y_pred,self.y_std = self.model.predict(self.df_X_pred) 

        if recalibrations == True:

            self.y_pred = np.zeros(self.df_X_pred.shape[0])
            self.y_std  = np.zeros(self.df_X_pred.shape[0])
            self.y_val  = self.df_y_val.values
            self.idx_recalibrations = []

            self.row = self.pred_start

            while self.row < self.pred_end:
                
                self.pred, self.std = self.model.predict(self.df_X_pred.iloc[[self.row-self.pred_start],:])

                if (np.abs(self.std.item()) > self.std_threshold) and \
                (np.abs(self.pred.item()-self.y_val[self.row-self.pred_start]) > self.e_threshold):
                    
                    print('Threshold exceeded at row ' + str(self.row))
                    self.idx_recalibrations.append(self.row)

                    self.df_X_retrain = self.df_X.iloc[0:self.row:self.N_skip,:]
                    self.df_y_retrain = self.df_y.iloc[0:self.row:self.N_skip,:]

                    self.model = mod.GPy_module()
                    self.model.train_model(self.df_X_retrain,self.df_y_retrain)

                    self.y_std[self.row-self.pred_start]  = 0
                    self.y_pred[self.row-self.pred_start] = self.y_val[self.row-self.pred_start]

                else:
                    
                    self.y_std[self.row-self.pred_start]  = self.std.item()
                    self.y_pred[self.row-self.pred_start] = self.pred.item()

                self.row = self.row + 1       

    def plot_result(self):

        self.y_resi = self.y_val-self.y_pred.reshape(-1,1)
        self.t = (self.df_X.index - self.df_X.index[0]).total_seconds()/3600
        self.t_pred = self.t[self.train_end+1:]
        self.y = self.df_y.values

        plt.style.use('default')
        plt.figure(figsize=(20, 8))
        plt.plot(self.t, self.y, label="Measurement")
        plt.plot(self.t_pred, self.y_pred, label="GP Model",color='red')
        plt.plot(self.t_pred, self.y_resi, label="Residual",color='grey')

        plt.fill_between(
            self.t_pred,
            self.y_pred-1.96*self.y_std,
            self.y_pred+1.96*self.y_std,
            color="tab:orange",
            alpha=0.5,
            label=r"95% Confidence Interval",
        )

        for x in self.idx_recalibrations:
            plt.axvline(x=t[x], color='black', linestyle='--')

        plt.legend(fontsize = 14)
        plt.xlabel('Time [h]',fontsize = 16)
        plt.ylabel('Error [μm]',fontsize = 16)
        plt.tick_params(axis='both',labelsize = 16)
        plt.axhline(0, color='black')  
        plt.grid()



