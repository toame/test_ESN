import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import math
#for path in ["NL_0_0.0_50_ring","NL_0_0.0_50_random","NL_0_0.0_100_ring","NL_0_0.0_100_random"]:
for path in ["NL_0_0.0_100_random2"]:
    NLtype = 'NL_old_cut1'
    for task in ["approx_3_0.0","approx_6_-0.5","approx_11_-1.0"]:
        df = pd.read_csv('C:\\Users\\tomo2\\Desktop\\test_ESN\\reservoir_ESN\\output_data\\' + path + '.csv', sep=',',comment='#');
        data_x = df['L']
        data_y = df[NLtype]
        value = df[task]
        plt.scatter(data_x, data_y,s=4, marker=".", c = value, alpha=1, linewidths=1, norm=LogNorm(vmin=0.05, vmax=1.0), cmap='viridis_r')
        plt.ylim(0.0, 200)
        plt.xlim(0.0, 50)
        #plt.legend(loc = "best")
        plt.colorbar()
        plt.savefig(path + "_" + task + "_" + NLtype + ".png", dpi = 600)
        plt.clf()
