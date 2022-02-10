import pandas as pd
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

file = "../output_data/NL_0_0.0_50_random2.csv"
df = pd.read_csv(file, sep=',',comment='#')
print(df.columns)
df = df[["p", "input_signal_factor", "bias_factor", "weight_factor", "L", "NL", "NL_old", "NL1_old", "narma3", "narma4", 
         "approx_3_1.5", "approx_6_1.0","approx_11_0.5" ]]
df = df[df["narma3"] <= 0.999]
df = df["narma3"]
#print(df)
corr_mat = df.corr(method='pearson')

print(corr_mat)
import seaborn as sns
sns.heatmap(corr_mat,
            vmin=-1.0,
            vmax=1.0,
            center=0,
            annot=True, # True:格子の中に値を表示
            fmt='.2f',
            xticklabels=corr_mat.columns.values,
            yticklabels=corr_mat.columns.values
           )
plt.show()
corr_mat.to_csv("test2.csv")
