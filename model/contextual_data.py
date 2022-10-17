import numpy as np
import torch

epsilon_q = 0.05


class ContextualLoop:
    def __init__(
        self,
        product_num,
        customer_num,
        q_upper,
        s_upper,
        lmbd_upper,
        seed,
        dim_feature=5,
        device="cpu",
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = device
        self.product_feat = torch.rand(product_num, dim_feature).to(device) / np.sqrt(
            dim_feature
        )
        self.customer_feat = (
            0.8 + 0.2 * torch.rand(customer_num, dim_feature).to(device)
        ) / np.sqrt(dim_feature)

        self.dim_customer = dim_feature
        self.dim_prod = dim_feature

        self.beta_q = torch.rand(dim_feature).to(device)
        self.beta_s = torch.rand(dim_feature).to(device)
        self.beta_lambda_product = torch.rand(dim_feature).to(device)
        self.beta_lambda_customer = torch.rand(dim_feature).to(device)

        self.reward = torch.rand(product_num).to(device)

        self.qs = torch.matmul(self.customer_feat, self.beta_q)
        self.ss = torch.matmul(self.customer_feat, self.beta_s)

        delta_q = q_upper / self.qs.max()
        self.qs = self.qs * delta_q
        self.beta_q = self.beta_q * delta_q

        delta_s = s_upper / self.ss.max()
        self.ss = self.ss * delta_s
        self.beta_s = self.beta_s * delta_s

        self.lambda_product = torch.matmul(self.product_feat, self.beta_lambda_product)
        self.lambda_customer = torch.matmul(
            self.customer_feat, self.beta_lambda_customer
        )

        delta_lmbd = lmbd_upper / (
            self.lambda_customer.max() + self.lambda_product.max()
        )
        self.lambda_customer = self.lambda_customer * delta_lmbd
        self.lambda_product = self.lambda_product * delta_lmbd
        self.beta_lambda_product = self.beta_lambda_product * delta_lmbd
        self.beta_lambda_customer = self.beta_lambda_customer * delta_lmbd

        self.product_num = product_num
        self.customer_num = customer_num

    def customer_step(self, ranking, t):
        q = self.qs[t]
        s = self.ss[t]
        lmbd = self.lambda_customer[t] + self.lambda_product
        lmbd = lmbd.cpu()

        browser_list = []
        purchase_list = []
        for i in range(self.product_num):
            k = ranking[i]
            browser_list.append(k)
            if np.random.rand() < lmbd[k]:
                purchase_list.append(1)
                if np.random.rand() > s * q:
                    break
            else:
                purchase_list.append(0)
                if np.random.rand() > q:
                    break
        return browser_list, purchase_list

    def expected_revenue(self, ranking, t):
        q = self.qs[t]
        s = self.ss[t]
        lmbd = self.lambda_customer[t] + self.lambda_product
        lmbd = lmbd[ranking]
        reward = self.reward[ranking]
        ps = torch.exp(torch.cumsum(torch.log(lmbd * q * s + (1 - lmbd) * q), dim=0))
        ps = torch.roll(ps, 1)
        ps[0] = 1
        revenue = torch.sum(ps * lmbd * reward).cpu()

        return revenue.item()
