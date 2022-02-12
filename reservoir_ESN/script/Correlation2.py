import pandas as pd
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

file = "../output_data/NL_0_0.0_50_random.csv"
df = pd.read_csv(file, sep=',',comment='#')
print(df.columns)
task =  "approx_6_-0.5"
df = df[["NL", "NL1_old",  task]]

df = df[df[task] <= 0.999]
df["NL"] = df["NL"] - df["NL"].mean() / df["NL"].std()
df["NL1_old"] = df["NL1_old"] - df["NL1_old"].mean() / df["NL1_old"].std()

#print(df)
# corr_mat = df.corr(method='pearson')

# print(corr_mat)
# import seaborn as sns
# sns.heatmap(corr_mat,
#             vmin=-1.0,
#             vmax=1.0,
#             center=0,
#             annot=True, # True:格子の中に値を表示
#             fmt='.2f',
#             xticklabels=corr_mat.columns.values,
#             yticklabels=corr_mat.columns.values
#            )
# plt.show()
# corr_mat.to_csv("test2.csv")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
import numpy as np

#統計検定である"alcohol"を除いたデータセットと、アルコールだけのデータセットを作成
df_X = df.drop(task, axis=1)
df_y = df[task]

#回帰
model = LinearRegression()
model.fit(df_X, df_y)

pred_y = model.predict(df_X)

print('決定係数（r2）:{}'.format(round(r2_score(df_y, pred_y),3)))
print('平均誤差（MAE）:{}'.format(round(mean_absolute_error(df_y, pred_y),3)))
print('RMSE:{}'.format(round(np.sqrt(mean_squared_error(df_y, pred_y)),3)))
"""output
決定係数（r2）:0.594
平均誤差（MAE）:0.408
RMSE:0.516
"""

#予測実測プロットの作成
plt.figure(figsize=(6,6))
plt.scatter(df_y, pred_y,color='blue',alpha=0.3)
y_max_ = max(df_y.max(), pred_y.max())
y_min_ = min(df_y.min(), pred_y.min())
y_max = y_max_ + (y_max_ - y_min_) * 0.1
y_min = y_min_ - (y_max_ - y_min_) * 0.1

plt.plot([y_min , y_max],[y_min, y_max], 'k-')

plt.ylim(y_min, y_max)
plt.xlim(y_min, y_max)
plt.xlabel('nmse(observed)',fontsize=20)
plt.ylabel('nmse(predicted)',fontsize=20)
plt.legend(loc='best',fontsize=15)
plt.title('yyplot(nmse)',fontsize=20)
plt.savefig('yyplot.png')
plt.show()

print('回帰係数：',round(model.intercept_,3))
"""output
回帰係数： 11.072
"""

#回帰係数を格納したpandasDataFrameの表示
df_coef =  pd.DataFrame({'coefficient':model.coef_.flatten()}, index=df_X.columns)

#グラフの作成
x_pos = np.arange(len(df_coef))

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(1, 1, 1)
ax1.barh(x_pos, df_coef['coefficient'], color='b')
ax1.set_title('coefficient of variables',fontsize=18)
ax1.set_yticks(x_pos)
ax1.set_yticks(np.arange(-1,len(df_coef.index))+0.5, minor=True)
ax1.set_yticklabels(df_coef.index, fontsize=14)
ax1.set_xticks(np.arange(-3,4,2)/10)
ax1.set_xticklabels(np.arange(-3,4,2)/10,fontsize=12)
ax1.grid(which='minor',axis='y',color='black',linestyle='-', linewidth=1)
ax1.grid(which='major',axis='x',linestyle='--', linewidth=1)
plt.savefig('coef_sklearn.png',bbox_inches='tight')
plt.show()
