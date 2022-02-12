import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import math
#for path in ["NL_0_0.0_50_ring","NL_0_0.0_50_random","NL_0_0.0_100_ring","NL_0_0.0_100_random"]:
for path in ["NL_0_0.0_100_random", "NL_0_0.0_100_ring"]:
    NLtype = "NL1_old"
    for task in ["approx_3_0.0","approx_6_-0.5","approx_11_-1.0"]:
        df = pd.read_csv('C:\\Users\\pc07-mis\\Desktop\\test_ESN\\reservoir_ESN\\output_data\\' + path + '_' + task + "_" + NLtype + '_average.csv', sep=',',comment='#');
        data_x = df['L'] + 1.0
        data_y = df['NL'] + 1.0
        value = df['nmse']
        plt.scatter(data_x, data_y,s=20, marker="s", c = value, alpha=1, linewidths=1, norm=LogNorm(vmin=0.05, vmax=1.0), cmap='viridis_r')
        plt.ylim(0.0, 100)
        plt.xlim(0.0, 100)
        #plt.legend(loc = "best")
        plt.colorbar()
        plt.savefig(path + "_" + task + "_" + NLtype + ".png", dpi = 600)
        plt.clf()
