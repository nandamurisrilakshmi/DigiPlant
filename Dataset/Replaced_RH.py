import numpy as np 
import pandas as pd








df=pd.read_csv("../Dataset/Earth_params/Photo100M_12JAN20 M2.csv")
dm=pd.read_csv("../Dataset/Mars_params/relative_humidity.csv")

df['col9'] = np.where(df['col9'] == 0, 
                      np.where(dm['col2'] == 0, df['col9']),
                      dm['col2'])
