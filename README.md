# MPB-UCB
Source code for NeurIPS 2022 paper [Product Ranking for Revenue Maximization with Multiple Purchases](https://arxiv.org/abs/2210.08268).

In this paper, we propose a more realistic consumer choice model to characterize consumer behaviors under multiple-purchase settings. We further develop the Multiple-Purchase-with-Budget UCB (MPB-UCB) algorithms with $\tilde{O}(\sqrt{T})$ regret that estimate consumers' behaviors and maximize revenue simultaneously in online settings.

## Installation
```bash
pip install -r requirements.txt
```

## Quick start
Take the non-contextual setting when $N=50$, $T=100,000$, $q=0.9$, $s=0.5$, $\lambda_{\max}=0.3$ as an example.
### Step 1: Calculate the optimal policy given full information
```bash
python main_non_contextual.py --method Optimal --num-prod 50 --num-consumer 100000 -q 0.9 -s 0.5 --lmbd-upper 0.3 --seed-parameter 666
```
### Step 2: Run our method with default hyper-parameters
```bash
python main_non_contextual.py --method Ours --num-prod 50 --num-consumer 100000 -q 0.9 -s 0.5 --lmbd-upper 0.3 --seed-parameter 666
```
Use `python main_non_contextual.py -h` to show all arguments for all baselines. The experiments are run 5 times with different seeds.

### Step 3: Plot the result
```bash
python plot.py --num-prod 50 --num-consumer 100000 -q 0.9 -s 0.5 --lmbd-upper 0.3 --seed-parameter 666
```
The figures on the regret, average revenue, revenue ratio are generated in the `figs/` directory.

### Grid search with [NNI](https://github.com/microsoft/nni)
Search the hyper-parameters of our method in the default setting.
```bash
nnictl create --config nni_ymls/config_non_contextual_Ours.yml --port 9000
```
Yamls for other baselines are included in the `nni_ymls/` directory.

## Citing MPB-UCB
```
@inproceedings{xu2022product,
    title={Product Ranking for Revenue Maximization with Multiple Purchases},
    author={Renzhe Xu and Xingxuan Zhang and Bo Li and Yafeng Zhang and Xiaolong Chen and Peng Cui},
    booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
    year={2022},
}
```
