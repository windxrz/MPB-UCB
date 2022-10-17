import numpy as np

epsilon_q = 0.05


class NonContextualLoop:
    def __init__(self, product_num, q, s, lmbd_upper, seed):
        np.random.seed(seed)
        self.true_qt = q
        self.true_st = s
        self.true_lambda = np.random.rand(product_num) * lmbd_upper
        self.reward = np.random.rand(product_num)
        self.product_num = product_num

    def consumer_step(self, ranking):
        """
        one step run for each consumer
        :return: a list of product purchased or not for a given consumer
                a list of browse or not
        """

        browser_list = []
        purchase_list = []
        for i in range(self.product_num):
            k = ranking[i]
            browser_list.append(k)
            if np.random.rand() < self.true_lambda[k]:
                purchase_list.append(k)
                if np.random.rand() > self.true_st * self.true_qt:
                    break
            else:
                if np.random.rand() > self.true_qt:
                    break
        return browser_list, purchase_list

    def expected_revenue(self, ranking):
        revenue = 0
        p = 1
        for i in range(self.product_num):
            revenue += p * self.true_lambda[ranking[i]] * self.reward[ranking[i]]
            p *= (
                self.true_lambda[ranking[i]] * self.true_qt * self.true_st
                + (1 - self.true_lambda[ranking[i]]) * self.true_qt
            )
        return revenue
