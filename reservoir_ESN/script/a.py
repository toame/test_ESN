import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('../output_data/add.csv')
print(df.head())
df = df.iloc[:10000,:]
df = df.loc[:,['NL_test', 'L_test', "NL_old", "L", "approx_3_0.0", "approx_6_-0.5", "approx_11_-1.0", "narma_5", "narma_10", "henon_3", "laser1_2", "count_5", "count_6", "laser0_2", "laser0_3", "laser1_3", "laser2_2", "laser2_3", "henon_4"]]
corr_mat = df.corr(method='pearson')
import seaborn as sons
sons.heatmap(corr_mat,
            vmin=-1.0,
            vmax=1.0,
            center=0,
            annot=True, # True:格子の中に値を表示
            fmt='.2f',
            xticklabels=corr_mat.columns.values,
            yticklabels=corr_mat.columns.values
           )
plt.show()