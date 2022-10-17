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

rc("text", usetex=True)

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42

linewidth = 3
markeredgewidth = 2
ms = 20
fontsize = 25
legend_fontsize = 20
label_fontsize = 25

COLOR_LIST = [
    "#C9625E",
    "#CBBD62",
    "#8ECB62",
    "#62CBC7",
    "#CD6ACA",
    "#000000",
    "#AAAAAA",
]
MODEL_LIST = [
    "Ours",
    "SinglePurchase",
    "KeepViewing",
    "ExploreThenExploitA",
    "ExploreThenExploitB",
]


def load_yaml(filename):
    with open(filename) as f:
        hyperparams = yaml.safe_load(f)
        f.close()
    return hyperparams


PLOT_RANGE = load_yaml("plot_range.yml")


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
    num_customer,
    num_prod,
    lmbd_upper,
    ax,
    type="regret",
):
    paras = [
        "seed_parameter_{}".format(seed_parameter),
        "q_{}".format(q),
        "s_{}".format(s),
        "num_customer_{}".format(num_customer),
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
        for k in range(5):
            revenue_list[k] = np.array(revenue_list[k])
        if type == "regret":
            for k in range(5):
                revenue_list[k] = bests - revenue_list[k]
        elif type == "revenue":
            for k in range(5):
                revenue_list[k] = revenue_list[k] / np.arange(
                    1, revenue_list[k].shape[0] * step + 1, step
                )
        elif type == "ratio":
            for k in range(5):
                revenue_list[k] = revenue_list[k] / bests

        if len(revenue_list) > 0:
            mean = np.mean(revenue_list, axis=0)
            std = np.std(revenue_list, axis=0)

            label = method.split("_")[0]
            label = label.replace("SinglePurchase", "Single Purchase")
            label = label.replace("KeepViewing", "Keep Viewing")
            label = label.replace("ExploreThenExploitA", "Explore Then Exploit A")
            label = label.replace("ExploreThenExploitB", "Explore Then Exploit B")

            linestyle = "-" if "Ours" in label else "--"
            ax.plot(
                range(0, num_customer, step),
                mean,
                label=label,
                color=COLOR_LIST[i],
                linestyle=linestyle,
                linewidth=linewidth,
            )
            ax.fill_between(
                range(0, num_customer, step),
                mean - std,
                mean + std,
                color=COLOR_LIST[i],
                alpha=0.1,
            )

            if (
                setting in PLOT_RANGE
                and params_setting in PLOT_RANGE[setting]
                and type in PLOT_RANGE[setting][params_setting]
            ):
                ymin = PLOT_RANGE[setting][params_setting][type][0]
                ymax = PLOT_RANGE[setting][params_setting][type][1]
                delta = (ymax - ymin) / 20
                ax.set_ylim(ymin - delta, ymax + delta)

    ax.set_title(
        "$N = {}$, $s{}={}$".format(
            num_prod,
            r"_{\max}" if "non" not in setting else "",
            s,
        ),
        fontsize=fontsize,
    )
    # ax.set_ylabel("Regret")


def plot_non_contextual(axes, num_prod, type, q):
    seed_parameter = 666
    num_customer = 100000
    lmbd_upper = 0.3
    for idx, s in enumerate([0.5, 0.8]):
        setting = "non_contextual"
        plot(
            setting,
            seed_parameter,
            q,
            s,
            num_customer,
            num_prod,
            lmbd_upper,
            axes[idx],
            type,
        )


def plot_contextual(axes, num_prod, type, q):
    seed_parameter = 666
    num_customer = 100000
    lmbd_upper = 0.3
    for idx, s in enumerate([0.5, 0.8]):
        setting = "contextual"
        plot(
            setting,
            seed_parameter,
            q,
            s,
            num_customer,
            num_prod,
            lmbd_upper,
            axes[idx],
            type,
        )


def plot_all(type, q):
    print("Plotting {}".format(type))
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.5))
    plot_non_contextual(axes[0, :2], 50, type, q)
    plot_non_contextual(axes[0, 2:], 300, type, q)
    plot_contextual(axes[1, :2], 50, type, q)
    plot_contextual(axes[1, 2:], 300, type, q)
    axes[0, 0].set_ylabel("Non-contextual", fontsize=fontsize)
    axes[1, 0].set_ylabel("Contextual", fontsize=fontsize)
    lines, labels = axes[0, 0].get_legend_handles_labels()
    if type == "regret":
        ytext = "Regret"
    elif type == "revenue":
        ytext = "Average revenue"
    elif type == "ratio":
        ytext = "Revenue ratio"
    fig.text(
        -0.03, 0.5, ytext, va="center", rotation="vertical", fontsize=fontsize * 1.2
    )
    fig.text(0.5, -0.04, r"\# of customers", ha="center", fontsize=fontsize * 1.2)
    fig.legend(
        lines,
        labels,
        prop={"size": legend_fontsize},
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=5,
    )
    plt.tight_layout()
    plt.savefig("figs/all_{}_q_{}.png".format(type, q), bbox_inches="tight")
    plt.savefig("figs/all_{}_q_{}.pdf".format(type, q), bbox_inches="tight")


def main():
    if not os.path.exists("figs"):
        os.mkdir("figs")
    for q in [0.9]:
        plot_all("regret", q)
        plot_all("revenue", q)
        plot_all("ratio", q)


if __name__ == "__main__":
    main()
