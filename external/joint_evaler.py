""" evaler.py """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np
import paddle as P
from tqdm import trange

from utils.log import logger as logging

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

        self.valid_config = config['eval']

    @P.no_grad()
    def run(self):
        '''
        print evaluation results
        '''
        self.model.eval()
        for eval_class in self.eval_classes_ser.values():
            eval_class.reset()
        for eval_class in self.eval_classes_link.values():
            eval_class.reset()

        total_time = 0.0
        total_frame = 0.0
        ser_macro_f1 = 0
        ser_micro_f1 = 0
        re_f1 = 0
        re_macro_f1 = 0
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
            table_results, ser_results = val.get_metric()
            metrics += '\n[Eval Validation - SER] {}:\n'.format(key) + str(table_results)
            ser_macro_f1 = ser_results['macro_f1']
            ser_micro_f1 = ser_results['micro_f1']
        for key, val in self.eval_classes_link.items():
            re_metrics = val.get_metric()
            metrics += '\n[Eval Validation - LINK] {}:\n'.format(key) + str(re_metrics)
            re_f1 = re_metrics['F1-SCORE']
            re_macro_f1 = re_metrics['Macro_f1']
        logging.info(metrics)

        return (ser_macro_f1, ser_micro_f1), (re_f1, re_macro_f1)

