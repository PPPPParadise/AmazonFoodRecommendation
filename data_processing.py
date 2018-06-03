import pandas as pd
import numpy as np
from interaction import *
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from loadDf import *
from processCabin import *
from processFare import *
from fill_n_scale import *
from ProcessName import *
from setMissingAges import *
from remove_corr import *

# Generates the data after feature engineering.
'''import functions and modify features'''


# load train, test data and concact into one --> df
# import function loadDF
train_df,test_df,df  = loadDf()

# import processCabin, processFamily, processPclass ...

processFare(df,keep_bins=1,keep_scaled=1)
processFamily(df,keep_scaled=1)
processEmbarked(df,keep_scaled=1)
processCabin(df,keep_scaled=1)
processName(df,keep_scaled=1,keep_bins=1,delName=1)
processPclass(df,keep_scaled=1)
processSex(df,keep_scaled=1)
processTicket(df,keep_scaled=1,keep_bins=1)
setMissingAges(df)

# Only keep scaled features
df1 = df.iloc[:,np.array([2,4,10,11,12,13,14,15,16,19,21,22,24,25,26,27,30])-1]

# interact.py --> create +, -, *, / between different features 
df2 = interact(df1)

# Export modified dataset to dfdf.csv
pd.DataFrame.to_csv(df2,'dfdf.csv',index=False)
df_uncorr = remove_corr(df2)

# New dataset after checking correlation < 0.9, export to df_uncorr.csv
pd.DataFrame.to_csv(df_uncorr,'df_uncorr.csv',index=False)