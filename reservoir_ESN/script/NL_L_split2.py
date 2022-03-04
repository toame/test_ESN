import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
#for path in ["NL_0_0.0_50_ring","NL_0_0.0_50_random","NL_0_0.0_100_ring","NL_0_0.0_100_random"]:
import argparse
parser = argparse.ArgumentParser(description='NL_L平面を描写する')
parser.add_argument('--root', default= "..//output_data//", help='ルートパスを指定する')
parser.add_argument('--output', default="..//fig//")
parser.add_argument('--data', default= ["NL_0_0.0_100_random"], help='ファイルを指定する', nargs='*')
parser.add_argument('--NL_types',default = ["NL_old"], help = "NL_typeを指定する", nargs='*')
parser.add_argument('--taskes',default=["approx_3_0.0", "approx_6_-0.5", "approx_11_-1.0", "narma10"], nargs='*')

args = parser.parse_args()
for path in args.data:
    for NL_type in args.NL_types:
        for task in args.taskes:
            df = pd.read_csv(args.root + path + '.csv', sep=',',comment='#')
            df = df[df["function_name"] == "sinc"]
            #df = df[df["bias_factor"] != 0]
            #df2.to_csv(task + ".csv")
            data_x = df['L_cut']
            data_y = df[NL_type]
            value = df[task]
            plt.scatter(data_x, data_y,s=3, marker="o", c = value, alpha=1, linewidths=1, norm=LogNorm(vmin=0.05, vmax=1.0), cmap='viridis_r')
            plt.ylim(0.0, 500)
            plt.xlim(0.0, 50)
            #plt.legend(loc = "best")
            plt.colorbar()
            plt.savefig(args.output + path + "_" + task + "_" + NL_type + ".png", dpi = 600)
            plt.clf()
