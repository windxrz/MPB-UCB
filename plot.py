import argparse
import copy
import json
import os

import matplotlib
import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib import rc
from tqdm import tqdm

rc("font", **{"family": "sans-serif", "sans-serif": ["Times New Roman"]})

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42

LINEWIDTH = 3
MARKEREDGEWIDTH = 2
MS = 20
FONTSIZE = 25
LEGEND_FONTSIZE = 20
LABEL_FONTSIZE = 25

COLOR_LIST = [
    "#CBBD62",
    "#8ECB62",
    "#62CBC7",
    "#CD6ACA",
    "#C9625E",
    "#000000",
    "#AAAAAA",
]
MODEL_LIST = [
    "SinglePurchase",
    "KeepViewing",
    "ExploreThenExploitA",
    "ExploreThenExploitB",
    "Ours",
]


def select_hyper(model_path, step=100):
    best_revenue = 0
    best_list = []
    best_hyperparams = ""
    for hyperparams in os.listdir(model_path):
        run_path = os.path.join(model_path, hyperparams)
        filename = os.path.join(run_path, "final.json")
        if not os.path.exists(filename):
            print(
                "Counting method {} with hyper parameters {}".format(
                    model_path.split("/")[-1], hyperparams
                )
            )
            count = 0
            tmp_list = []
            for seed in os.listdir(run_path):
                if "final" in seed:
                    continue
                with open(os.path.join(run_path, seed)) as f:
                    info = json.loads(f.read())
                    f.close()
                revenues = np.array(info["revenues"])
                revenues = np.cumsum(revenues)[::step].tolist()
                tmp_list.append(revenues)
                count += 1
            if len(tmp_list) == 0:
                continue
            if count < 5 and not "Optimal" in model_path:
                continue
            res = {}
            revenue = np.mean(tmp_list, axis=0)[-1]
            res["all"] = tmp_list
            res["revenue"] = revenue
            with open(filename, "w") as f:
                f.write(json.dumps(res))
                f.close()
        else:
            with open(filename) as f:
                res = json.loads(f.read())
                f.close()
        if res["revenue"] > best_revenue:
            best_revenue = res["revenue"]
            best_list = res["all"]
            best_hyperparams = hyperparams

    return best_list, best_hyperparams


def plot(
    setting,
    seed_parameter,
    q,
    s,
    num_consumer,
    num_prod,
    lmbd_upper,
    ax,
    type="regret",
):
    paras = [
        "seed_parameter_{}".format(seed_parameter),
        "q_{}".format(q),
        "s_{}".format(s),
        "num_consumer_{}".format(num_consumer),
        "num_prod_{}".format(num_prod),
        "lmbd_upper_{}".format(lmbd_upper),
    ]

    ans_path = os.path.join("results", setting)
    if not os.path.exists(ans_path):
        return

    flag = False
    params_setting = ""
    for params_setting in os.listdir(ans_path):
        flag = True
        for para in paras:
            if not para in params_setting:
                flag = False
                break
        if flag:
            ans_path = os.path.join(ans_path, params_setting)
            break
    if not flag:
        return

    optimal_file = os.path.join(ans_path, "Optimal", "no_hyperparams", "0.json")
    with open(optimal_file) as f:
        info = json.loads(f.read())
        f.close()

    step = 100
    bests = np.array(info["revenues"])
    bests = np.cumsum(bests)
    bests = bests[::step]

    for i, method in enumerate(tqdm(MODEL_LIST)):
        method_path = os.path.join(ans_path, method)
        if not os.path.exists(method_path):
            continue
        revenue_list, hyperparams = select_hyper(method_path, step)
        n_method = len(revenue_list)
        for k in range(n_method):
            revenue_list[k] = np.array(revenue_list[k])
        if type == "regret":
            for k in range(n_method):
                revenue_list[k] = bests - revenue_list[k]
        elif type == "average revenue":
            for k in range(n_method):
                revenue_list[k] = revenue_list[k] / np.arange(
                    1, revenue_list[k].shape[0] * step + 1, step
                )
        elif type == "revenue ratio":
            for k in range(n_method):
                revenue_list[k] = revenue_list[k] / bests

        if len(revenue_list) > 0:
            mean = np.mean(revenue_list, axis=0)
            std = np.std(revenue_list, axis=0)

            label = method.split("_")[0]
            label = label.replace("SinglePurchase", "Single Purchase")
            label = label.replace("KeepViewing", "Keep Viewing")
            label = label.replace("ExploreThenExploitA", "Explore Then Exploit A")
            label = label.replace("ExploreThenExploitB", "Explore Then Exploit B")
            label = label.replace("Ours", "MPB-UCB (Ours)")

            linestyle = "-" if "Ours" in label else "--"
            ax.plot(
                range(0, num_consumer, step),
                mean,
                label=label,
                color=COLOR_LIST[i],
                linestyle=linestyle,
                linewidth=LINEWIDTH,
            )
            ax.fill_between(
                range(0, num_consumer, step),
                mean - std,
                mean + std,
                color=COLOR_LIST[i],
                alpha=0.1,
            )

            if type == "revenue ratio":
                ax.set_ylim(0.948, 1.002)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting",
        type=str,
        choices=["contextual", "non_contextual"],
        default="non_contextual",
    )
    parser.add_argument("--num-prod", type=int, default=50)
    parser.add_argument("--num-consumer", type=int, default=100000)
    parser.add_argument("-q", type=float, default=0.9)
    parser.add_argument("-s", type=float, default=0.5)
    parser.add_argument("--lmbd-upper", type=float, default=0.3)
    parser.add_argument("--seed-parameter", type=int, default=666)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    setting = args.setting
    params = (
        "num_prod_{}_num_consumer_{}_q_{}_s_{}_lmbd_upper_{}_seed_parameter_{}".format(
            args.num_prod,
            args.num_consumer,
            args.q,
            args.s,
            args.lmbd_upper,
            args.seed_parameter,
        )
    )
    if not os.path.exists("figs"):
        os.mkdir("figs")
    print(setting, params)
    output_path = os.path.join("results", setting, params)
    if not os.path.exists(output_path):
        print("No results in this setting!")
        exit()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, type in enumerate(["regret", "average revenue", "revenue ratio"]):
        plot(
            setting,
            args.seed_parameter,
            args.q,
            args.s,
            args.num_consumer,
            args.num_prod,
            args.lmbd_upper,
            axes[i],
            type,
        )
        axes[i].set_title(type.capitalize(), fontsize=FONTSIZE)
    fig.text(0.5, -0.04, r"# of consumers", ha="center", fontsize=FONTSIZE * 1.2)
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        lines,
        labels,
        prop={"size": LEGEND_FONTSIZE},
        loc="lower center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=5,
    )
    fig.tight_layout()
    plt.savefig("figs/{}_{}.png".format(setting, params), bbox_inches="tight")


if __name__ == "__main__":
    main()
