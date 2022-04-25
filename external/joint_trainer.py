""" trainer.py """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np
import paddle as P
import paddle.fluid as fluid
from tqdm import trange
from .losses import RELoss
from paddle.nn import CrossEntropyLoss

from utils.log import logger as logging

class JointTrainer:
    """
    Trainer class
    """

    def __init__(self, config, model, data_loader):
        '''
        :param config:
        :param model:
        :param data_loader:
        '''
        self.model = model
        self.train_data_loader = data_loader
        self.len_step = len(self.train_data_loader)
        self.config = config
        self.train_config = config['train']
        self.ser_loss = CrossEntropyLoss()
        self.link_loss_fn = RELoss(alpha = self.train_config['loss']['loss_bce'],
                                   beta = self.train_config['loss']['loss_rank'])
        t_total = len(self.train_data_loader) * self.train_config['epoch']

        self.lr_scheduler = P.optimizer.lr.PolynomialDecay(
            learning_rate = self.train_config['optimizer']['lr'],
            decay_steps = t_total,
            end_lr = 0.0,
            power=1.0
        )

        if self.train_config['optimizer']['warmup_steps'] > 0:
            self.lr_scheduler = P.optimizer.lr.LinearWarmup(
                self.lr_scheduler,
                self.train_config['optimizer']['warmup_steps'],
                start_lr = 0,
                end_lr=self.train_config['optimizer']['lr']
            )
        grad_clip = P.nn.ClipGradByNorm(clip_norm=10)
        self.optimizer = P.optimizer.Adam(
            learning_rate = self.train_config['optimizer']['lr'],
            parameters = self.model.parameters(),
            epsilon = self.train_config['optimizer']['epsilon'],
            grad_clip = grad_clip,
            weight_decay = self.train_config['optimizer']['weight_decay']
        )
        self.model.train()

    def run_epoch(self):
        total_time = 0.0
        total_frame = 0.0
        total_loss = []
        t = trange(self.len_step)
        loader = self.train_data_loader()
        for step_idx in t:
            t.set_description('train with example %i' % step_idx)
            input_data = next(loader)
            start = time.time()
            feed_names = self.train_config['feed_names']

            output = self.model(*input_data, feed_names=feed_names)
            ser_output = output['labeling_segment']
            link_output = output['linking']

            ser_loss = self.ser_loss(ser_output['logit'], ser_output['label'])
            link_loss = self.link_loss_fn(link_output['logit'], link_output['label'])
            loss = ser_loss + link_loss
            total_loss.append(loss.item())
            loss.backward()

            self.optimizer.step()
            self.optimizer.clear_grad()
            total_time += time.time() - start
            total_frame += input_data[0].shape[0]
            # if step_idx % self.config['monitoring']['log_iter'] == 0:
            #     logging.info("iter [{}/{}]: loss: {:0.6f}, lr: {:0.6f}, total_time: {:0.6f}".format(step_idx, self.len_step, np.mean(total_loss), self.optimizer.get_lr(), total_time))

        return np.mean(total_loss), self.optimizer.get_lr(), total_time

    def train(self):
        return self.run_epoch()

