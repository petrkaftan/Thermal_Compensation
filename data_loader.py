import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

class Measurement:

    def __init__(self,dataType,testNumber,testDate,machine):

        global_directory = 'C:/Users/pkaftan/polybox'

        self.machine    = machine
        self.dataType   = dataType
        self.testNumber = testNumber
        self.testDate   = testDate

        self.testFolderName = 'Test' + str(testNumber) + '_' + testDate + '/02 Results'
        self.dir  = global_directory + r"/000 " + machine + r"/000 Measurement Data/" + self.testFolderName 

    def read_file(self):
        
        # Read file
        self.df = pd.read_csv(self.dir + "/" + self.testDate + '_Log_' + self.dataType + '.csv',encoding='unicode_escape')
        
        # Convert time information from text to datetime format
        formats =  ['%d.%m.%Y %H:%M:%S.%f', '%Y.%m.%d %H:%M:%S.%f']
        for fmt in formats:
            try: 
                self.df.iloc[:,0] = pd.to_datetime(self.df.iloc[:,0], format = fmt)
                break
            except ValueError:
                continue

        # Set datetime column to index
        self.df.index = pd.DatetimeIndex(self.df['Date/Time'])
        self.df = self.df.drop(columns=['Date/Time'])
        # Drop the rows where all elements are missing
        self.df = self.df.dropna(how='all')
        # Drop columns with zero values
        self.df = self.df.loc[:, (self.df != 0).any(axis=0)]
        # Drop columns with nans
        self.df = self.df.dropna(axis='columns',how='all')

    def zero_normalize(self):
        self.df = self.df - self.df.iloc[0]

    def select_sensors(self,sensor_names):
        self.df = self.df.loc[:,sensor_names]

    def remove_sensors(self,sensor_names):
        self.df = self.df.drop(sensor_names,axis=1)

    def get_global_time(self):
        global_time = self.df.sort_index().loc[~self.df.index.duplicated(keep='first')].index
        return global_time
    
    def get_t_delta(self):
        t_delta = (self.df.index - self.df.index[0]).total_seconds()/3600
        return t_delta 

    def synchronize_time(self,global_time):
        # augment = self augmented with interpolated values with time indices of global_time
        augment = pd.DataFrame(np.NaN, index=pd.to_datetime(global_time), columns=self.df.columns)
        augment = self.df.combine_first(augment).interpolate('time')

        # unique = unique time indices from augment compared to self
        unique = augment[~augment.index.isin(self.df.index)]

        # missing = missing indices in unique compared to global_time
        missing = global_time.to_frame()[~global_time.isin(unique.index)]

        # result = concatenate unique and missing indices from self
        result = pd.concat([unique,self.df.loc[missing.index]])   

        # Remove duplicate indexes
        result = result[~result.index.duplicated(keep='first')]

        # Sort
        result = result.sort_index()
    
        # Attach to object
        self.df = result

    def get_correlation(self,target_df):
        
        corr_df = (
            target_df.corrwith(self.df, method='pearson')
            .to_frame(name='Correlation')          # make it a DataFrame
            .reset_index()                         # bring the index into a column
            .rename(columns={'index': 'Feature'})  # rename for clarity
        )
        print(corr_df)

    def smooth(self, window_length=11, polyorder=3):

        if isinstance(self.df, pd.Series):
            # single Series
            filtered = savgol_filter(self.df.values, window_length=window_length, polyorder=polyorder)
            # rebuild as Series to preserve index & name
            self.df = pd.Series(filtered, index=self.df.index, name=self.df.name)

        elif isinstance(self.df, pd.DataFrame):
            # DataFrame: apply column‐wise in place
            for col in self.df.columns:
                self.df[col] = savgol_filter(
                    self.df[col].values,
                    window_length=window_length,
                    polyorder=polyorder
                )
        else:
            raise TypeError(f"Unsupported type for self.df: {type(self.df)}")

    def load_measurement(self,error_name=None,zero_normalize=False,global_time=None,remove_sensors=None):

        self.read_file()

        if error_name:
            self.select_sensors(error_name)
        
        if global_time is not None:
            self.synchronize_time(global_time)

        # Normalization AFTER time synchronization!
        if zero_normalize == True:
            self.zero_normalize()

        if remove_sensors:
            self.remove_sensors(remove_sensors)

        print('Loading' + ' ' + self.dataType + ' finished.')