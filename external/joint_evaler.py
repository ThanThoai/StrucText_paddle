""" evaler.py """
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

class JointEvaler:
    """
    Evaler class
    """

    def __init__(self, config, model, data_loader, eval_classes_ser=None, eval_classes_link=None):
        '''
        :param config:
        :param model:
        :param data_loader:
        '''
        self.model = model
        self.eval_classes_ser = eval_classes_ser
        self.eval_classes_link = eval_classes_link
        self.valid_data_loader = data_loader
        self.len_step = len(self.valid_data_loader)

        self.init_model = config['init_model']
        self.valid_config = config['eval']

    @P.no_grad()
    def run(self):
        '''
        print evaluation results
        '''
        self._resume_model()
        self.model.eval()
        for eval_class in self.eval_classes_ser.values():
            eval_class.reset()
        for eval_class in self.eval_classes_link.values():
            eval_class.reset()

        total_time = 0.0
        total_frame = 0.0
        t = trange(self.len_step)
        loader = self.valid_data_loader()
        for step_idx in t:
            t.set_description('evaluate with example %i' % step_idx)
            input_data = next(loader)
            start = time.time()
            feed_names = self.valid_config['feed_names']
            output = self.model(*input_data, feed_names=feed_names)
            ser_output = output['labeling_segment']
            link_output = output['linking']
            total_time += time.time() - start

            ####### Eval SER ##########
            for key, val in self.eval_classes_ser.items():
                if 'entity' in key and 'label_prim' in output.keys():
                    label = ser_output['label_prim'].numpy()
                    pred = ser_output['pred_prim'].numpy()
                    mask = output.get('mask', None)
                else:
                    # print(ser_output)
                    label = ser_output['label'].numpy()
                    pred = ser_output['pred'].numpy()
                    mask = ser_output.get('mask', None)
                mask = None if mask is None else mask.numpy()
                val(pred, label, mask)
            #########################

            ####### Eval LINK ##########
            for key, val in self.eval_classes_link.items():
                # print(link_output)
                label = link_output['label'].numpy()
                logit = link_output['logit'].numpy()
                mask = None
                val(logit, label, mask)
            #########################
            total_frame += input_data[0].shape[0]
        metrics = 'fps : {}'.format(total_frame / total_time)
        for key, val in self.eval_classes_ser.items():
            metrics += '\n[Eval Validation - SER] {}:\n'.format(key) + str(val.get_metric())
        for key, val in self.eval_classes_link.items():
            metrics += '\n[Eval Validation - LINK] {}:\n'.format(key) + str(val.get_metric())
        print(metrics)

    def _resume_model(self):
        '''
        Resume from saved model
        :return:
        '''
        para_path = self.init_model
        if para_path != None and os.path.exists(para_path):
            para_dict = P.load(para_path)
            self.model.set_dict(para_dict)
            logging.info('Load init model from %s', para_path)
        else:
            logging.info('Checkpoint is not found')