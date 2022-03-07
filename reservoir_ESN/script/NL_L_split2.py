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
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 5, 1)
            ax2 = fig.add_subplot(1, 5, 2)
            ax3 = fig.add_subplot(1, 5, 3)
            ax4 = fig.add_subplot(1, 5, 4)
            ax5 = fig.add_subplot(1, 5, 5)
            df = pd.read_csv(args.root + path + '.csv', sep=',',comment='#')
            for num, function_name, bias, ax in [(1, "all", True, ax1), (2, "sinc", True, ax2), (3, "tanh", True, ax3), (4, "sinc", False, ax4),(5, "tanh", False, ax5)]:
                print(path, NL_type, task)
                df2 = df[df["function_name"] == function_name]
                if bias:
                    df2 = df2[df2["bias_factor"] != 0]
                else:
                    df2 = df2[df2["bias_factor"] == 0]
                #df2.to_csv(task + ".csv")
                if function_name == "all":
                    df2 = df
                print(df2)
                data_x = df2['L_cut']
                if NL_type == "NL_test":
                    data_y = df2["NL_old_2"] + df2["NL_old_3"] + df2["NL_old_4"] + df2["NL_old_5"] + df2["NL_old_6"]
                else:
                    data_y = df2[NL_type]
                value = df2[task]
                mappable = ax.scatter(data_x, data_y,s=1, marker=".", c = value, alpha=1, linewidths=1, norm=LogNorm(vmin=0.05, vmax=1.0), cmap='viridis_r')
                if num != 1:
                    ax.tick_params(axis = 'y', labelcolor = "None")
                ax.set_ylim(0.0, 500)
                ax.set_xlim(0.0, 50)
                ax.set_title(function_name + "," + str(bias))
                #plt.legend(loc = "best")
            fig.colorbar(mappable, ax=ax)
            fig.savefig(args.output + path + "_" + task + "_" + NL_type + ".png", dpi = 600)
            fig.clf()
