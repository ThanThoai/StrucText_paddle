import paddle
import paddle.nn as nn
import paddle.tensor as tensor
import paddle.nn.functional as F
from paddle.nn import Layer
from paddle.nn import BCELoss, MarginRankingLoss
import math

import numpy as np

class MarginRankingRELoss(nn.Layer):

    def __init__(self, margin = 0):
        super().__init__()
        self.margin = margin

    def forward(self, logits, labels, margin = 0):
        loss_rank = 0.0
        bs, w, h = logits.shape
        y = 0
        for b in range(bs):
            logit_flatten = paddle.fluid.layers.nn.flatten(logits[b], axis = 0).numpy()[0].reshape(-1, 1)
            logit_flatten = logit_flatten @ np.ones(np.shape(logit_flatten)[::-1])

            logit_flatten = logit_flatten.T - logit_flatten
            y = np.zeros(np.shape(logit_flatten))
            label = labels[b].numpy()
            h_l, w_l = np.shape(label)
            h, w = np.shape(y)
            for i in range(h):
                for j in range(w):
                    if label[i // h_l][i % w_l] == 1.0 and label[j // h_l][j % w_l] == 0.0:
                        y[i][j] = 1
            
            output = -y * logit_flatten + self.margin
            output[output <= 0] = 0.0
            loss_rank += np.sum(output)
        return loss_rank


class RELoss(nn.Layer):

    def __init__(self, alpha = 1, beta = 1):
        super(RELoss, self).__init__()
        self.loss_bce = BCELoss()
        # self.loss_rank = MarginRankingRELoss()
        self.alpha = 1
        self.beta = 1

    def forward(self, logits, labels):
        loss_bce = 0.0
        loss_rank = 0.0
        # for b in range(batchsize):
        loss_bce += self.loss_bce(logits, labels)
        # loss_rank += self.loss_rank(link_logit, labels)
        # print(loss_bce)
        # print(loss_rank)
        # total_loss = self.alpha * loss_bce + self.beta * loss_rank
        return loss_bce
        # return total_loss, loss_bce, loss_rank




