""" trainer.py """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import logging
import numpy as np
import paddle as P
from tqdm import trange
from .linking.loss import RELoss
import paddle.fluid as fluid
import paddle

class Trainer:
    """
    Trainer class
    """

    def __init__(self, config, model, data_loader, logging):
        '''
        :param config:
        :param model:
        :param data_loader:
        '''
        self.model = model
        self.train_data_loader = data_loader
        self.len_step = len(self.train_data_loader)
        self.config = config
        self.init_model = config['init_model']
        self.train_config = config['train']
        self.loss_fn = RELoss(alpha = self.train_config['loss']['loss_bce'], beta = self.train_config['loss']['loss_rank'])
        self.logging = logging
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
        self._resume_model()

    def run_epoch(self):
        '''

        '''

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
            # loss, loss_bce, loss_rank = self.loss_fn(output)
            loss = self.loss_fn(output)
            total_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()
            total_time += time.time() - start
            total_frame += input_data[0].shape[0]
            if step_idx % self.config['monitoring']['log_iter'] == 0:
                self.logging.info("iter [{}/{}]: loss: {:0.6f}, lr: {:0.6f}, total_time: {:0.6f}".format(step_idx, self.len_step, np.mean(total_loss), self.optimizer.get_lr(), total_time))
        
    def train(self):
        for epoch in range(self.train_config['epoch']):
            self.logging.info(f"Training in epoch {epoch}/{self.train_config['epoch']}")
            # self.run_epoch()
            if epoch % self.config['monitoring']['save_module'] == 0:
                # self.model.save_pretrained(self.config['monitoring']['save_dir'])

                # self.model.(self.config['monitoring']['save_dir'])
                path_model = os.path.join(self.config['monitoring']['save_dir'], f"epoch_{epoch}.params")
                paddle.save(self.model.state_dict(), path_model)

    def _resume_model(self):
        '''
        Resume from saved model
        :return:
        '''
        para_path = self.init_model
        if para_path != None and os.path.exists(para_path):
            para_dict = P.load(para_path)
            self.model.set_dict(para_dict)
            self.logging.info('Load init model from %s', para_path)
        else:
            self.logging.info('Checkpoint is not found')
