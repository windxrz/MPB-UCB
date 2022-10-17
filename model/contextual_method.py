import numpy as np
import torch

from model.contextual_data import ContextualLoop, epsilon_q


def rank_products(lmbd, r, q, s):
    tmp = r * lmbd / (1 - q + q * (1 - s) * lmbd)
    idx = torch.argsort(tmp, descending=True)
    return idx.cpu()


class Optimal:
    def __init__(self, product_num: int, customer_num: int, loop: ContextualLoop):
        self.name = "Optimal"
        self.product_num = product_num
        self.customer_num = customer_num
        self.reward = loop.reward

        self.lmbd_product = loop.lambda_product
        self.lmbd_customer = loop.lambda_customer
        self.qs = loop.qs
        self.ss = loop.ss

    def policy(self, t):
        lmbd = self.lmbd_customer[t] + self.lmbd_product
        q = self.qs[t]
        s = self.ss[t]
        ans = rank_products(lmbd, self.reward, q, s)
        return ans

    def feedback(self, browser_list, purchase_list, t):
        pass


class SingleOptimal:
    def __init__(self, product_num: int, customer_num: int, loop: ContextualLoop):
        self.name = "SingleOptimal"
        self.product_num = product_num
        self.customer_num = customer_num
        self.reward = loop.reward

        self.lmbd_product = loop.lambda_product
        self.lmbd_customer = loop.lambda_customer
        self.qs = loop.qs
        self.ss = loop.ss

    def policy(self, t):
        lmbd = self.lmbd_customer[t] + self.lmbd_product
        q = self.qs[t]
        s = self.ss[t]
        tmp = self.reward * lmbd / (1 - q + q * lmbd)
        idx = torch.argsort(tmp, descending=True)
        return idx

    def feedback(self, browser_list, purchase_list, t):
        pass


class KeepOptimal:
    def __init__(self, product_num: int, customer_num: int, loop: ContextualLoop):
        self.name = "KeepOptimal"
        self.product_num = product_num
        self.customer_num = customer_num
        self.reward = loop.reward

        self.lmbd_product = loop.lambda_product
        self.lmbd_customer = loop.lambda_customer
        self.qs = loop.qs
        self.ss = loop.ss

    def policy(self, t):
        lmbd = self.lmbd_customer[t] + self.lmbd_product
        q = self.qs[t]
        s = self.ss[t]
        tmp = self.reward * lmbd / (1 - q - (1 - q) * lmbd + 1e-5)
        idx = torch.argsort(tmp, descending=True)
        return idx

    def feedback(self, browser_list, purchase_list, t):
        pass


class Ours:
    def __init__(
        self,
        product_num: int,
        customer_num: int,
        loop: ContextualLoop,
        alpha: float,
        xiq: float,
        xiw: float,
        xilmbd: float,
    ):
        self.name = "Ours"
        self.device = loop.device
        self.product_num = product_num
        self.customer_num = customer_num
        self.product_feat = loop.product_feat
        self.customer_feat = loop.customer_feat
        self.reward = loop.reward
        self.dim_customer = loop.dim_customer
        self.dim_prod = loop.dim_prod

        self.alpha = alpha
        self.xiq = xiq
        self.xiw = xiw
        self.xilmbd = xilmbd

        self.sigma_lmbd = self.alpha * torch.eye(self.dim_customer + self.dim_prod).to(
            self.device
        )
        self.rho_lmbd = torch.zeros(self.dim_customer + self.dim_prod).to(self.device)
        self.sigma_q = self.alpha * torch.eye(self.dim_customer).to(self.device)
        self.rho_q = torch.zeros(self.dim_customer).to(self.device)
        self.sigma_w = (
            self.alpha * 10 * torch.eye(self.dim_customer ** 2).to(self.device)
        )
        self.rho_w = torch.zeros(self.dim_customer ** 2).to(self.device)

        self.qs = loop.qs
        self.ss = loop.ss
        self.true_beta_q = loop.beta_q
        self.true_beta_s = loop.beta_s
        self.true_beta_lmbd = torch.cat(
            [loop.beta_lambda_customer, loop.beta_lambda_product], dim=0
        )

    def tau(self, t, m, alpha):
        return 1 / 2 * np.sqrt(
            m * np.log(1 + t * self.product_num / m / alpha) + 4 * np.log(t)
        ) + np.sqrt(alpha)

    def policy(self, t):
        inv_lmbd = torch.linalg.inv(self.sigma_lmbd)
        inv_q = torch.linalg.inv(self.sigma_q)
        inv_w = torch.linalg.inv(self.sigma_w)
        beta_lmbd = torch.matmul(inv_lmbd, self.rho_lmbd)
        beta_q = torch.matmul(inv_q, self.rho_q)
        beta_w = torch.matmul(inv_w, self.rho_w)
        tau_lmbd = self.tau(t + 1, self.dim_customer + self.dim_prod, self.alpha)
        tau_q = self.tau(t + 1, self.dim_customer, self.alpha)
        tau_w = self.tau(t + 1, self.dim_customer ** 2, self.alpha)

        q_feat = self.customer_feat[t]
        w_feat = self.customer_feat[t]
        w_feat = torch.matmul(w_feat.view(-1, 1), w_feat.view(1, -1)).view(-1)

        q = (
            torch.matmul(beta_q, q_feat)
            + self.xiq
            * tau_q
            * torch.sqrt(
                torch.matmul(
                    torch.matmul(q_feat.view(1, -1), inv_q), q_feat.view(-1, 1)
                )
            )[0][0]
        )
        q = torch.clamp(q, 0, 1 - epsilon_q)

        w = (
            torch.matmul(beta_w, w_feat)
            + self.xiw
            * tau_w
            * torch.sqrt(
                torch.matmul(
                    torch.matmul(w_feat.view(1, -1), inv_w), w_feat.view(-1, 1)
                )
            )[0][0]
        )
        w = torch.clamp(w, 0, q)

        lmbd_feat = torch.cat(
            [
                self.customer_feat[t]
                .view(-1, self.dim_prod)
                .expand(self.product_num, -1),
                self.product_feat,
            ],
            dim=1,
        )
        lmbd = torch.matmul(lmbd_feat, beta_lmbd) + self.xilmbd * tau_lmbd * torch.sqrt(
            torch.sum((torch.matmul(lmbd_feat, inv_lmbd) * lmbd_feat), dim=1)
        )

        s = (w / q) if q > 0 else 0

        ans = rank_products(lmbd, self.reward, q, s)
        return ans

    def feedback(self, browser_list, purchase_list, t):
        for i in range(len(browser_list)):
            k = browser_list[i]
            p = purchase_list[i]
            cust_prod_feat = torch.cat(
                [self.customer_feat[t], self.product_feat[k]], dim=0
            )
            tmp = torch.matmul(cust_prod_feat.view(-1, 1), cust_prod_feat.view(1, -1))
            self.sigma_lmbd += tmp
            self.rho_lmbd += cust_prod_feat * p
            if p == 0:
                cust_feat = self.customer_feat[t]
                self.sigma_q += torch.matmul(
                    cust_feat.view(-1, 1), cust_feat.view(1, -1)
                )
                self.rho_q += (i < len(browser_list) - 1) * cust_feat
            else:
                feat = self.customer_feat[t]
                feat = torch.matmul(feat.view(-1, 1), feat.view(1, -1)).view(-1)
                self.sigma_w += torch.matmul(feat.view(-1, 1), feat.view(1, -1))
                self.rho_w += (i < len(browser_list) - 1) * feat


class SinglePurchase:
    def __init__(
        self,
        product_num: int,
        customer_num: int,
        loop: ContextualLoop,
        alpha: float,
        xiq: float,
        xilmbd: float,
    ):
        self.name = "SinglePurchase"
        self.product_num = product_num
        self.customer_num = customer_num
        self.product_feat = loop.product_feat
        self.customer_feat = loop.customer_feat
        self.reward = loop.reward
        self.dim_customer = loop.dim_customer
        self.dim_prod = loop.dim_prod
        self.device = loop.device

        self.alpha = alpha
        self.xiq = xiq
        self.xilmbd = xilmbd

        self.sigma_lmbd = self.alpha * torch.eye(self.dim_customer + self.dim_prod).to(
            self.device
        )
        self.rho_lmbd = torch.zeros(self.dim_customer + self.dim_prod).to(self.device)
        self.sigma_q = self.alpha * torch.eye(self.dim_customer).to(self.device)
        self.rho_q = torch.zeros(self.dim_customer).to(self.device)

        self.qs = loop.qs
        self.ss = loop.ss

    def tau(self, t, m, alpha):
        return 1 / 2 * np.sqrt(
            m * np.log(1 + t * self.product_num / m / alpha) + 4 * np.log(t)
        ) + np.sqrt(alpha)

    def policy(self, t):
        inv_lmbd = torch.linalg.inv(self.sigma_lmbd)
        inv_q = torch.linalg.inv(self.sigma_q)
        beta_lmbd = torch.matmul(inv_lmbd, self.rho_lmbd)
        beta_q = torch.matmul(inv_q, self.rho_q)
        tau_lmbd = self.tau(t + 1, self.dim_customer + self.dim_prod, self.alpha)
        tau_q = self.tau(t + 1, self.dim_customer, self.alpha)

        q_feat = self.customer_feat[t]

        q = (
            torch.matmul(beta_q, q_feat)
            + self.xiq
            * tau_q
            * torch.sqrt(
                torch.matmul(
                    torch.matmul(q_feat.view(1, -1), inv_q), q_feat.view(-1, 1)
                )
            )[0][0]
        )
        q = torch.clamp(q, 0, 1 - epsilon_q)

        lmbd_feat = torch.cat(
            [
                self.customer_feat[t]
                .view(-1, self.dim_prod)
                .expand(self.product_num, -1),
                self.product_feat,
            ],
            dim=1,
        )
        lmbd = torch.matmul(lmbd_feat, beta_lmbd) + self.xilmbd * tau_lmbd * torch.sqrt(
            torch.sum((torch.matmul(lmbd_feat, inv_lmbd) * lmbd_feat), dim=1)
        )

        s = 0

        return rank_products(lmbd, self.reward, q, s)

    def feedback(self, browser_list, purchase_list, t):
        for i in range(len(browser_list)):
            k = browser_list[i]
            p = purchase_list[i]
            cust_prod_feat = torch.cat(
                [self.customer_feat[t], self.product_feat[k]], dim=0
            )
            tmp = torch.matmul(cust_prod_feat.view(-1, 1), cust_prod_feat.view(1, -1))
            self.sigma_lmbd += tmp
            self.rho_lmbd += cust_prod_feat * p
            if p == 0:
                cust_feat = self.customer_feat[t]
                self.sigma_q += torch.matmul(
                    cust_feat.view(-1, 1), cust_feat.view(1, -1)
                )
                self.rho_q += (i < len(browser_list) - 1) * cust_feat


class KeepViewing:
    def __init__(
        self,
        product_num: int,
        customer_num: int,
        loop: ContextualLoop,
        alpha: float,
        xiq: float,
        xilmbd: float,
    ):
        self.name = "KeepViewing"
        self.product_num = product_num
        self.customer_num = customer_num
        self.product_feat = loop.product_feat
        self.customer_feat = loop.customer_feat
        self.reward = loop.reward
        self.dim_customer = loop.dim_customer
        self.dim_prod = loop.dim_prod
        self.device = loop.device

        self.alpha = alpha
        self.xiq = xiq
        self.xilmbd = xilmbd

        self.sigma_lmbd = self.alpha * torch.eye(self.dim_customer + self.dim_prod).to(
            self.device
        )
        self.rho_lmbd = torch.zeros(self.dim_customer + self.dim_prod).to(self.device)
        self.sigma_q = self.alpha * torch.eye(self.dim_customer).to(self.device)
        self.rho_q = torch.zeros(self.dim_customer).to(self.device)

        self.qs = loop.qs
        self.ss = loop.ss
        self.true_beta_q = loop.beta_q
        self.true_beta_s = loop.beta_s
        self.true_beta_lmbd = torch.cat(
            [loop.beta_lambda_customer, loop.beta_lambda_product], dim=0
        )

    def tau(self, t, m, alpha):
        return 1 / 2 * np.sqrt(
            m * np.log(1 + t * self.product_num / m / alpha) + 4 * np.log(t)
        ) + np.sqrt(alpha)

    def policy(self, t):
        inv_lmbd = torch.linalg.inv(self.sigma_lmbd)
        inv_q = torch.linalg.inv(self.sigma_q)
        beta_lmbd = torch.matmul(inv_lmbd, self.rho_lmbd)
        beta_q = torch.matmul(inv_q, self.rho_q)
        tau_lmbd = self.tau(t + 1, self.dim_customer + self.dim_prod, self.alpha)
        tau_q = self.tau(t + 1, self.dim_customer, self.alpha)

        q_feat = self.customer_feat[t]

        q = (
            torch.matmul(beta_q, q_feat)
            + self.xiq
            * tau_q
            * torch.sqrt(
                torch.matmul(
                    torch.matmul(q_feat.view(1, -1), inv_q), q_feat.view(-1, 1)
                )
            )[0][0]
        )
        q = torch.clamp(q, 0, 1 - epsilon_q)

        lmbd_feat = torch.cat(
            [
                self.customer_feat[t]
                .view(-1, self.dim_prod)
                .expand(self.product_num, -1),
                self.product_feat,
            ],
            dim=1,
        )
        lmbd = torch.matmul(lmbd_feat, beta_lmbd) + self.xilmbd * tau_lmbd * torch.sqrt(
            torch.sum((torch.matmul(lmbd_feat, inv_lmbd) * lmbd_feat), dim=1)
        )

        tmp = self.reward * lmbd / (1 - q - (1 - q) * lmbd + 1e-5)
        idx = torch.argsort(tmp, descending=True)
        return idx

    def feedback(self, browser_list, purchase_list, t):
        for i in range(len(browser_list)):
            k = browser_list[i]
            p = purchase_list[i]
            cust_prod_feat = torch.cat(
                [self.customer_feat[t], self.product_feat[k]], dim=0
            )
            tmp = torch.matmul(cust_prod_feat.view(-1, 1), cust_prod_feat.view(1, -1))
            self.sigma_lmbd += tmp
            self.rho_lmbd += cust_prod_feat * p
            if p == 0:
                cust_feat = self.customer_feat[t]
                self.sigma_q += torch.matmul(
                    cust_feat.view(-1, 1), cust_feat.view(1, -1)
                )
                self.rho_q += (i < len(browser_list) - 1) * cust_feat
            else:
                pass


class ExploreThenExploitA:
    def __init__(
        self,
        product_num: int,
        customer_num: int,
        loop: ContextualLoop,
        alpha: float,
        delta: float,
    ):
        self.name = "ExploreThenExploitA"
        self.product_num = product_num
        self.customer_num = customer_num
        self.product_feat = loop.product_feat
        self.customer_feat = loop.customer_feat
        self.reward = loop.reward
        self.alpha = alpha
        self.dim_customer = loop.dim_customer
        self.device = loop.device

        self.dim_prod = loop.dim_prod
        self.delta = delta
        self.lower_count = int(self.delta * np.log(self.customer_num))

        self.sigma_lmbd = self.alpha * torch.eye(self.dim_customer + self.dim_prod).to(
            self.device
        )
        self.rho_lmbd = torch.zeros(self.dim_customer + self.dim_prod).to(self.device)
        self.sigma_q = self.alpha * torch.eye(self.dim_customer).to(self.device)
        self.rho_q = torch.zeros(self.dim_customer).to(self.device)
        self.sigma_w = (
            self.alpha * 10 * torch.eye(self.dim_customer ** 2).to(self.device)
        )
        self.rho_w = torch.zeros(self.dim_customer ** 2).to(self.device)

        self.browse_count_list = torch.zeros(product_num, dtype=torch.int)

        self.qs = loop.qs
        self.ss = loop.ss
        self.true_beta_q = loop.beta_q
        self.true_beta_s = loop.beta_s
        self.true_beta_lmbd = torch.cat(
            [loop.beta_lambda_customer, loop.beta_lambda_product], dim=0
        )

    def tau(self, t, m, alpha):
        return 1 / 2 * np.sqrt(
            m * np.log(1 + t * self.product_num / m / alpha) + 4 * np.log(t)
        ) + np.sqrt(alpha)

    def policy(self, t):
        tmp = torch.clamp(self.lower_count - self.browse_count_list, min=0)
        if torch.max(tmp).item() == 0:
            inv_lmbd = torch.linalg.inv(self.sigma_lmbd)
            inv_q = torch.linalg.inv(self.sigma_q)
            inv_w = torch.linalg.inv(self.sigma_w)
            beta_lmbd = torch.matmul(inv_lmbd, self.rho_lmbd)
            beta_q = torch.matmul(inv_q, self.rho_q)
            beta_w = torch.matmul(inv_w, self.rho_w)

            q_feat = self.customer_feat[t]
            w_feat = self.customer_feat[t]
            w_feat = torch.matmul(w_feat.view(-1, 1), w_feat.view(1, -1)).view(-1)

            q = torch.matmul(beta_q, q_feat)
            q = torch.clamp(q, 0, 1 - epsilon_q)

            w = torch.matmul(beta_w, w_feat)
            w = torch.clamp(w, 0, q)

            r = self.reward
            lmbd_feat = torch.cat(
                [
                    self.customer_feat[t]
                    .view(-1, self.dim_prod)
                    .expand(self.product_num, -1),
                    self.product_feat,
                ],
                dim=1,
            )

            lmbd = torch.matmul(lmbd_feat, beta_lmbd).cpu()
            r = self.reward.cpu()

            s = w / (q + 1e-5)

            return rank_products(lmbd, r, q.cpu(), s.cpu())
        else:
            idx = torch.argsort(tmp, descending=True)
            return idx

    def feedback(self, browser_list, purchase_list, t):
        for i in range(len(browser_list)):
            k = browser_list[i]
            self.browse_count_list[k] += 1
            p = purchase_list[i]
            cust_prod_feat = torch.cat(
                [self.customer_feat[t], self.product_feat[k]], dim=0
            )
            tmp = torch.matmul(cust_prod_feat.view(-1, 1), cust_prod_feat.view(1, -1))
            self.sigma_lmbd += tmp
            self.rho_lmbd += cust_prod_feat * p
            if p == 0:
                cust_feat = self.customer_feat[t]
                self.sigma_q += torch.matmul(
                    cust_feat.view(-1, 1), cust_feat.view(1, -1)
                )
                self.rho_q += (i < len(browser_list) - 1) * cust_feat
            else:
                feat = self.customer_feat[t]
                feat = torch.matmul(feat.view(-1, 1), feat.view(1, -1)).view(-1)
                self.sigma_w += torch.matmul(feat.view(-1, 1), feat.view(1, -1))
                self.rho_w += (i < len(browser_list) - 1) * feat


class ExploreThenExploitB:
    def __init__(
        self,
        product_num: int,
        customer_num: int,
        loop: ContextualLoop,
        alpha: float,
        delta: float,
    ):
        self.name = "ExploreThenExploitB"
        self.product_num = product_num
        self.customer_num = customer_num
        self.product_feat = loop.product_feat
        self.customer_feat = loop.customer_feat
        self.reward = loop.reward
        self.alpha = alpha
        self.dim_customer = loop.dim_customer
        self.dim_prod = loop.dim_prod
        self.delta = delta
        self.lower_count = int(self.delta * np.log(self.customer_num))
        self.device = loop.device

        self.sigma_lmbd = self.alpha * torch.eye(self.dim_customer + self.dim_prod).to(
            self.device
        )
        self.rho_lmbd = torch.zeros(self.dim_customer + self.dim_prod).to(self.device)
        self.sigma_q = self.alpha * torch.eye(self.dim_customer).to(self.device)
        self.rho_q = torch.zeros(self.dim_customer).to(self.device)
        self.sigma_w = (
            self.alpha * 10 * torch.eye(self.dim_customer ** 2).to(self.device)
        )
        self.rho_w = torch.zeros(self.dim_customer ** 2).to(self.device)

        self.browse_count_list = torch.zeros(product_num)

        self.qs = loop.qs
        self.ss = loop.ss
        self.true_beta_q = loop.beta_q
        self.true_beta_s = loop.beta_s
        self.true_beta_lmbd = torch.cat(
            [loop.beta_lambda_customer, loop.beta_lambda_product], dim=0
        )

    def tau(self, t, m, alpha):
        return 1 / 2 * np.sqrt(
            m * np.log(1 + t * self.product_num / m / alpha) + 4 * np.log(t)
        ) + np.sqrt(alpha)

    def policy(self, t):
        inv_lmbd = torch.linalg.inv(self.sigma_lmbd)
        inv_q = torch.linalg.inv(self.sigma_q)
        inv_w = torch.linalg.inv(self.sigma_w)
        beta_lmbd = torch.matmul(inv_lmbd, self.rho_lmbd)
        beta_q = torch.matmul(inv_q, self.rho_q)
        beta_w = torch.matmul(inv_w, self.rho_w)

        q_feat = self.customer_feat[t]
        w_feat = self.customer_feat[t]
        w_feat = torch.matmul(w_feat.view(-1, 1), w_feat.view(1, -1)).view(-1)

        q = torch.matmul(beta_q, q_feat)
        q = torch.clamp(q, 0, 1 - epsilon_q)

        w = torch.matmul(beta_w, w_feat)
        w = torch.clamp(w, 0, q)

        r = torch.clone(self.reward)

        lmbd_feat = torch.cat(
            [
                self.customer_feat[t]
                .view(-1, self.dim_prod)
                .expand(self.product_num, -1),
                self.product_feat,
            ],
            dim=1,
        )

        lmbd = torch.matmul(lmbd_feat, beta_lmbd).cpu()

        lmbd = torch.clamp(lmbd, 0, 1)
        lmbd = torch.where(
            self.browse_count_list < self.lower_count, torch.tensor(1.0), lmbd
        )
        r = torch.where(
            self.browse_count_list < self.lower_count,
            torch.tensor(1e3) * (self.lower_count - self.browse_count_list),
            r,
        )

        s = (w / q) if q > 0 else 0

        return rank_products(lmbd, r, q, s)

    def feedback(self, browser_list, purchase_list, t):
        for i in range(len(browser_list)):
            k = browser_list[i]
            self.browse_count_list[k] += 1
            p = purchase_list[i]
            cust_prod_feat = torch.cat(
                [self.customer_feat[t], self.product_feat[k]], dim=0
            )
            tmp = torch.matmul(cust_prod_feat.view(-1, 1), cust_prod_feat.view(1, -1))
            self.sigma_lmbd += tmp
            self.rho_lmbd += cust_prod_feat * p
            if p == 0:
                cust_feat = self.customer_feat[t]
                self.sigma_q += torch.matmul(
                    cust_feat.view(-1, 1), cust_feat.view(1, -1)
                )
                self.rho_q += (i < len(browser_list) - 1) * cust_feat
            else:
                feat = self.customer_feat[t]
                feat = torch.matmul(feat.view(-1, 1), feat.view(1, -1)).view(-1)
                self.sigma_w += torch.matmul(feat.view(-1, 1), feat.view(1, -1))
                self.rho_w += (i < len(browser_list) - 1) * feat
