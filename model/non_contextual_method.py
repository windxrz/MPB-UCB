import numpy as np

from model.non_contextual_data import NonContextualLoop, epsilon_q


def rank_products(lmbd, r, q, s):
    tmp = r * lmbd / (1 - q + q * (1 - s) * lmbd)
    idx = np.argsort(tmp)[::-1]
    return idx


class Optimal:
    def __init__(self, product_num: int, consumer_num: int, loop: NonContextualLoop):
        self.name = "Optimal"
        self.product_num = product_num
        self.consumer_num = consumer_num
        self.reward = loop.reward
        self.lmbd = loop.true_lambda
        self.q = loop.true_qt
        self.s = loop.true_st

    def policy(self, t):
        ans = rank_products(self.lmbd, self.reward, self.q, self.s)
        return ans

    def feedback(self, browser_list, purchase_list):
        pass


class SingleOptimal:
    def __init__(self, product_num: int, consumer_num: int, loop: NonContextualLoop):
        self.name = "SingleOptimal"
        self.product_num = product_num
        self.consumer_num = consumer_num
        self.reward = loop.reward
        self.lmbd = loop.true_lambda
        self.q = loop.true_qt
        self.s = loop.true_st

    def policy(self, t):
        ans = rank_products(self.lmbd, self.reward, self.q, 0)
        return ans

    def feedback(self, browser_list, purchase_list):
        pass


class KeepOptimal:
    def __init__(self, product_num: int, consumer_num: int, loop: NonContextualLoop):
        self.name = "KeepOptimal"
        self.product_num = product_num
        self.consumer_num = consumer_num
        self.reward = loop.reward
        self.lmbd = loop.true_lambda
        self.q = loop.true_qt
        self.s = loop.true_st

    def policy(self, t):
        tmp = self.reward * self.lmbd / (1 - self.q - (1 - self.q) * self.lmbd + 1e-5)
        idx = np.argsort(tmp)[::-1]
        return idx

    def feedback(self, browser_list, purchase_list):
        pass


class Ours:
    def __init__(
        self,
        product_num: int,
        consumer_num: int,
        loop: NonContextualLoop,
        xiq: float,
        xiw: float,
        xilmbd: float,
    ):
        self.name = "Ours"
        self.product_num = product_num
        self.consumer_num = consumer_num
        self.reward = loop.reward
        self.xiq = xiq
        self.xiw = xiw
        self.xilmbd = xilmbd

        self.purchase_num = 0
        self.browse_not_pur_num = 0
        self.continue_browse_no_purchase_num = 0
        self.continue_browse_after_pur_num = 0

        self.browse_count_list = np.zeros(product_num)
        self.purchase_count_list = np.zeros(product_num)

    def policy(self, t):
        if self.browse_not_pur_num == 0:
            q = 1 - epsilon_q
        else:
            q = (
                self.continue_browse_no_purchase_num / self.browse_not_pur_num
                + self.xiq * np.sqrt(np.log(t) / self.browse_not_pur_num)
            )
            q = min(q, 1 - epsilon_q)

        if self.purchase_num == 0:
            w = q
        else:
            w = (
                self.continue_browse_after_pur_num / self.purchase_num
                + self.xiw * np.sqrt(np.log(t) / self.purchase_num)
            )
            w = min(w, q)

        s = (w / q) if q > 0 else 0

        lmbd = np.zeros(self.product_num)
        for k in range(self.product_num):
            if self.browse_count_list[k] == 0:
                lmbd[k] = 1
            else:
                lmbd[k] = (
                    self.purchase_count_list[k] / self.browse_count_list[k]
                    + self.xilmbd * np.sqrt(np.log(t) / self.browse_count_list[k]) / 2
                )
                lmbd[k] = min(lmbd[k], 1)

        ans = rank_products(lmbd, self.reward, q, s)
        return ans

    def feedback(self, browser_list, purchase_list):
        self.purchase_num += len(purchase_list)
        self.continue_browse_after_pur_num += len(purchase_list)
        self.browse_not_pur_num += len(browser_list) - len(purchase_list)
        self.continue_browse_no_purchase_num += len(browser_list) - len(purchase_list)
        if len(browser_list) < self.product_num:
            if len(purchase_list) > 0 and browser_list[-1] == purchase_list[-1]:
                self.continue_browse_after_pur_num -= 1
            else:
                self.continue_browse_no_purchase_num -= 1

        for t in browser_list:
            self.browse_count_list[t] += 1
        for t in purchase_list:
            self.purchase_count_list[t] += 1


class SinglePurchase:
    def __init__(
        self,
        product_num: int,
        consumer_num: int,
        loop: NonContextualLoop,
        xiq: float,
        xilmbd: float,
    ):
        self.name = "SinglePurchase"
        self.product_num = product_num
        self.consumer_num = consumer_num
        self.reward = loop.reward

        self.browse_not_pur_num = 0
        self.continue_browse_no_purchase_num = 0

        self.browse_count_list = np.zeros(product_num)
        self.purchase_count_list = np.zeros(product_num)
        self.xiq = xiq
        self.xilmbd = xilmbd

    def policy(self, t):
        if self.browse_not_pur_num == 0:
            q = 1 - epsilon_q
        else:
            q = (
                self.continue_browse_no_purchase_num / self.browse_not_pur_num
                + self.xiq * np.sqrt(np.log(t) / self.browse_not_pur_num)
            )
            q = min(q, 1 - epsilon_q)

        lmbd = np.zeros(self.product_num)
        for k in range(self.product_num):
            if self.browse_count_list[k] == 0:
                lmbd[k] = 1
            else:
                lmbd[k] = self.purchase_count_list[k] / self.browse_count_list[
                    k
                ] + self.xilmbd * np.sqrt(np.log(t) / self.browse_count_list[k])
                lmbd[k] = min(lmbd[k], 1)

        return rank_products(lmbd, self.reward, q, 0)

    def feedback(self, browser_list, purchase_list):
        self.browse_not_pur_num += len(browser_list) - len(purchase_list)
        self.continue_browse_no_purchase_num += len(browser_list) - len(purchase_list)
        if len(browser_list) < self.product_num:
            if len(purchase_list) > 0 and browser_list[-1] == purchase_list[-1]:
                pass
            else:
                self.continue_browse_no_purchase_num -= 1

        for t in browser_list:
            self.browse_count_list[t] += 1
        for t in purchase_list:
            self.purchase_count_list[t] += 1


class KeepViewing:
    def __init__(
        self,
        product_num: int,
        consumer_num: int,
        loop: NonContextualLoop,
        xiq: float,
        xilmbd: float,
    ):
        self.name = "KeepViewing"
        self.product_num = product_num
        self.consumer_num = consumer_num
        self.reward = loop.reward
        self.xiq = xiq
        self.xilmbd = xilmbd

        self.browse_not_pur_num = 0
        self.continue_browse_no_purchase_num = 0

        self.browse_count_list = np.zeros(product_num)
        self.purchase_count_list = np.zeros(product_num)

    def policy(self, t):
        if self.browse_not_pur_num == 0:
            q = 1 - epsilon_q
        else:
            q = (
                self.continue_browse_no_purchase_num / self.browse_not_pur_num
                + self.xiq * np.sqrt(np.log(t) / self.browse_not_pur_num)
            )
            q = min(q, 1 - epsilon_q)

        lmbd = np.zeros(self.product_num)
        for k in range(self.product_num):
            if self.browse_count_list[k] == 0:
                lmbd[k] = 1
            else:
                lmbd[k] = self.purchase_count_list[k] / self.browse_count_list[
                    k
                ] + self.xilmbd * np.sqrt(np.log(t) / self.browse_count_list[k])
                lmbd[k] = min(lmbd[k], 1)

        tmp = self.reward * lmbd / (1 - q - (1 - q) * lmbd + 1e-5)
        idx = np.argsort(tmp)[::-1]
        return idx

    def feedback(self, browser_list, purchase_list):
        self.browse_not_pur_num += len(browser_list) - len(purchase_list)
        self.continue_browse_no_purchase_num += len(browser_list) - len(purchase_list)
        if len(browser_list) < self.product_num:
            if len(purchase_list) > 0 and browser_list[-1] == purchase_list[-1]:
                pass
            else:
                self.continue_browse_no_purchase_num -= 1

        for t in browser_list:
            self.browse_count_list[t] += 1
        for t in purchase_list:
            self.purchase_count_list[t] += 1


class ExploreThenExploitA:
    def __init__(
        self, product_num: int, consumer_num: int, loop: NonContextualLoop, delta: float
    ):
        self.name = "ExploreThenExploitA"
        self.product_num = product_num
        self.consumer_num = consumer_num
        self.delta = delta

        self.r = loop.reward

        self.lower_count = int(self.delta * np.log(self.consumer_num))

        self.purchase_num = 0
        self.browse_not_pur_num = 0
        self.continue_browse_no_purchase_num = 0
        self.continue_browse_after_pur_num = 0

        self.browse_count_list = np.zeros(product_num)
        self.purchase_count_list = np.zeros(product_num)

        self.ranking = None

    def policy(self, t):
        tmp = np.maximum(self.lower_count - self.browse_count_list, 0)
        if np.max(tmp) == 0:
            q = min(
                1 - epsilon_q,
                self.continue_browse_no_purchase_num / self.browse_not_pur_num,
            )
            w = min(q, self.continue_browse_after_pur_num / self.purchase_num)
            s = w / q
            lmbd = np.zeros(self.product_num)
            for i in range(self.product_num):
                lmbd[i] = self.purchase_count_list[i] / self.browse_count_list[i]
            return rank_products(lmbd, self.r, q, s)
        else:
            idx = np.argsort(tmp)[::-1]
            return idx

    def feedback(self, browser_list, purchase_list):
        self.purchase_num += len(purchase_list)
        self.continue_browse_after_pur_num += len(purchase_list)
        self.browse_not_pur_num += len(browser_list) - len(purchase_list)
        self.continue_browse_no_purchase_num += len(browser_list) - len(purchase_list)
        if len(browser_list) < self.product_num:
            if len(purchase_list) > 0 and browser_list[-1] == purchase_list[-1]:
                self.continue_browse_after_pur_num -= 1
            else:
                self.continue_browse_no_purchase_num -= 1

        for t in browser_list:
            self.browse_count_list[t] += 1
        for t in purchase_list:
            self.purchase_count_list[t] += 1


class ExploreThenExploitB:
    def __init__(
        self, product_num: int, consumer_num: int, loop: NonContextualLoop, delta: float
    ):
        self.name = "ExploreThenExploitB"
        self.product_num = product_num
        self.consumer_num = consumer_num
        self.delta = delta

        self.r = loop.reward

        self.lower_count = int(self.delta * np.log(self.consumer_num))

        self.purchase_num = 0
        self.browse_not_pur_num = 0
        self.continue_browse_no_purchase_num = 0
        self.continue_browse_after_pur_num = 0

        self.browse_count_list = np.zeros(product_num)
        self.purchase_count_list = np.zeros(product_num)

        self.ranking = None

    def policy(self, t):
        tmp = np.maximum(self.lower_count - self.browse_count_list, 0)
        if np.max(tmp) == 0:
            q = min(
                1 - epsilon_q,
                self.continue_browse_no_purchase_num / self.browse_not_pur_num,
            )
            w = min(q, self.continue_browse_after_pur_num / self.purchase_num)
            s = w / q
            lmbd = np.zeros(self.product_num)
            for i in range(self.product_num):
                lmbd[i] = self.purchase_count_list[i] / self.browse_count_list[i]
            return rank_products(lmbd, self.r, q, s)
        else:
            r = np.copy(self.r)
            if self.browse_not_pur_num == 0:
                q = 1 - epsilon_q
            else:
                q = min(
                    1 - epsilon_q,
                    self.continue_browse_no_purchase_num / self.browse_not_pur_num,
                )
            if self.purchase_num == 0:
                w = q
            else:
                w = min(q, self.continue_browse_after_pur_num / self.purchase_num)
            if q == 0:
                s = 1
            else:
                s = w / q
            lmbd = np.zeros(self.product_num)
            for i in range(self.product_num):
                if self.browse_count_list[i] < self.lower_count:
                    lmbd[i] = 1
                    r[i] = 1e5 * tmp[i]
                else:
                    lmbd[i] = self.purchase_count_list[i] / self.browse_count_list[i]
            return rank_products(lmbd, r, q, s)

    def feedback(self, browser_list, purchase_list):
        self.purchase_num += len(purchase_list)
        self.continue_browse_after_pur_num += len(purchase_list)
        self.browse_not_pur_num += len(browser_list) - len(purchase_list)
        self.continue_browse_no_purchase_num += len(browser_list) - len(purchase_list)
        if len(browser_list) < self.product_num:
            if len(purchase_list) > 0 and browser_list[-1] == purchase_list[-1]:
                self.continue_browse_after_pur_num -= 1
            else:
                self.continue_browse_no_purchase_num -= 1

        for t in browser_list:
            self.browse_count_list[t] += 1
        for t in purchase_list:
            self.purchase_count_list[t] += 1
