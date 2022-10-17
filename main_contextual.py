import argparse
import json
import os

import nni
import numpy as np
import torch
from tqdm import tqdm

from model.contextual_data import ContextualLoop
from model.contextual_method import (
    ExploreThenExploitA,
    ExploreThenExploitB,
    KeepOptimal,
    KeepViewing,
    Optimal,
    Ours,
    SingleOptimal,
    SinglePurchase,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prod", type=int, default=300)
    parser.add_argument("--num-consumer", type=int, default=100000)
    parser.add_argument("-q", type=float, default=0.9)
    parser.add_argument("-s", type=float, default=0.5)
    parser.add_argument("--lmbd-upper", type=float, default=0.3)
    parser.add_argument("--seed-parameter", type=int, default=666)

    parser.add_argument(
        "--method",
        choices=[
            "Ours",
            "SinglePurchase",
            "KeepViewing",
            "ExploreThenExploitA",
            "ExploreThenExploitB",
            "Optimal",
            "SingleOptimal",
            "KeepOptimal",
        ],
        default="Ours",
    )

    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--xiq", type=float, default=0.2)
    parser.add_argument("--xiw", type=float, default=0.05)
    parser.add_argument("--xilmbd", type=float, default=0.1)

    parser.add_argument("--nni", action="store_true")
    parser.add_argument("--gpu", action="store_true")

    args = parser.parse_args()
    return args


def main():
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists("results/contextual"):
        os.mkdir("results/contextual")

    args = parse_args()
    if args.nni:
        my_dict = vars(args)
        optimized_params = nni.get_next_parameter()
        my_dict.update(optimized_params)
        args = argparse.Namespace(**my_dict)

    print(args)

    setting = (
        "num_prod_{}_num_consumer_{}_q_{}_s_{}_lmbd_upper_{}_seed_parameter_{}".format(
            args.num_prod,
            args.num_consumer,
            args.q,
            args.s,
            args.lmbd_upper,
            args.seed_parameter,
        )
    )
    method_name = args.method
    if "Optimal" in args.method:
        hyper_params = "no_hyperparams"
    elif args.method in [
        "ExploreThenExploitA",
        "ExploreThenExploitB",
    ]:
        hyper_params = "delta_{}".format(args.delta)
    else:
        hyper_params = "alpha_{}_xiq_{}_{}xilmbd_{}".format(
            args.alpha,
            args.xiq,
            "xiw_{}_".format(args.xiw) if "Ours" in args.method else "",
            args.xilmbd,
        )

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "contextual",
        setting,
        method_name,
        hyper_params,
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ans_revenue = []
    device = "cuda" if args.gpu else "cpu"
    for seed_consumer in range(1 if "Optimal" in args.method else 5):
        filename = os.path.join(output_path, "{}.json".format(seed_consumer))
        if not os.path.exists(filename):
            loop = ContextualLoop(
                args.num_prod,
                args.num_consumer,
                args.q,
                args.s,
                args.lmbd_upper,
                args.seed_parameter,
                device=device,
            )

            if args.method == "Ours":
                method = Ours(
                    args.num_prod,
                    args.num_consumer,
                    loop,
                    args.alpha,
                    args.xiq,
                    args.xiw,
                    args.xilmbd,
                )
            elif args.method == "KeepViewing":
                method = KeepViewing(
                    args.num_prod,
                    args.num_consumer,
                    loop,
                    args.alpha,
                    args.xiq,
                    args.xilmbd,
                )
            elif args.method == "SinglePurchase":
                method = SinglePurchase(
                    args.num_prod,
                    args.num_consumer,
                    loop,
                    args.alpha,
                    args.xiq,
                    args.xilmbd,
                )
            elif args.method == "ExploreThenExploitA":
                method = ExploreThenExploitA(
                    args.num_prod,
                    args.num_consumer,
                    loop,
                    args.alpha,
                    args.delta,
                )
            elif args.method == "ExploreThenExploitB":
                method = ExploreThenExploitB(
                    args.num_prod,
                    args.num_consumer,
                    loop,
                    args.alpha,
                    args.delta,
                )
            elif args.method == "Optimal":
                method = Optimal(args.num_prod, args.num_consumer, loop)
            elif args.method == "SingleOptimal":
                method = SingleOptimal(args.num_prod, args.num_consumer, loop)
            elif args.method == "KeepOptimal":
                method = KeepOptimal(args.num_prod, args.num_consumer, loop)

            np.random.seed(seed_consumer)
            torch.manual_seed(seed_consumer)
            revenues = []
            for t in tqdm(range(args.num_consumer)):
                ranking = method.policy(t)
                revenue = loop.expected_revenue(ranking, t)
                revenues.append(revenue)
                browser_list, purchase_list = loop.consumer_step(ranking, t)
                method.feedback(browser_list, purchase_list, t)

            ans = {
                "revenues": revenues,
            }

            with open(filename, "w") as f:
                f.write(json.dumps(ans))
                f.close()
        else:
            with open(filename) as f:
                ans = json.loads(f.read())
                f.close()
            revenues = ans["revenues"]

        ans_revenue.append(np.sum(revenues))

    if args.nni:
        report = {"default": np.mean(ans_revenue), "std": np.std(ans_revenue)}
        nni.report_final_result(report)


if __name__ == "__main__":
    main()
