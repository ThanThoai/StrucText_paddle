""" eval_infer.py """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import sys
import json
import argparse
import functools
import importlib
import numpy as np
import paddle as P

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from utils.utility import add_arguments, print_arguments
from utils.build_dataloader import build_dataloader
from utils.metrics import build_metric
from utils.log import logger as logging
from external.joint_evaler import JointEvaler
from external.joint_trainer import JointTrainer

np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# sysconf
# base
parser = argparse.ArgumentParser('launch for eval')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--task_type', type=str, required=True)
parser.add_argument('--weights_path', type=str, default=None)

args = parser.parse_args()
print_arguments(args)
config = json.loads(open(args.config_file).read())

ALL_MODULES = ['joint']
if args.task_type not in ALL_MODULES:
	raise ValueError('Not valid task_type %s in %s' % (args.task_type, str(ALL_MODULES)))

# modules
model = importlib.import_module('external.' + args.task_type + '.modules.model')
dataset = importlib.import_module('external.' + args.task_type + '.dataset')

# package_name
Model = model.Model
Dataset = dataset.Dataset

def main(config):
	# program
	train_config = config['train']
	eval_config = config['eval']
	model_config = config['architecture']

	weights_path = args.weights_path

	config['init_model'] = weights_path
	train_config['dataset']['max_seqlen'] = \
			model_config['embedding']['max_seqlen']
	eval_config['dataset']['max_seqlen'] = \
			model_config['embedding']['max_seqlen']

	place = P.set_device('cpu')

	train_dataset = Dataset(
			train_config['dataset'],
			train_config['feed_names'],
			True
	)

	train_loader = build_dataloader(
			config['train'], 
			train_dataset,
			"Train",
			place,
			False
	)

	eval_dataset = Dataset(
			eval_config['dataset'],
			eval_config['feed_names'],
			False
	)

	eval_loader = build_dataloader(
			config['eval'],
			eval_dataset,
			'Eval',
			place,
			False)
	#model
	model = Model(model_config, train_config['feed_names'])

	#metric
	eval_classes_ser = build_metric(eval_config['ser_metric'])
	eval_classes_link = build_metric(eval_config['link_metric'])
	# best metrics
	best_ser_macro_f1 = 0
	best_ser_micro_f1 = 0
	best_re_f1 = 0
	best_re_macro_f1 = 0
	should_save_ckpt = False

	trainer = JointTrainer(config, model, train_loader)
	evaler = JointEvaler(config, model, eval_loader, eval_classes_ser, eval_classes_link)

	logging.info("Start training........................")
	for epoch in range(train_config['epoch']):
		logging.info(f"Training in epoch {epoch}/{train_config['epoch']}")
		trainer.train()
		logging.info(f"Evaluation in epoch {epoch}/{train_config['epoch']}")
		(ser_macro_f1, ser_micro_f1), (re_f1, re_macro_f1) = evaler.run()

		if ser_macro_f1 > best_ser_macro_f1:
			best_ser_macro_f1 = ser_macro_f1
			should_save_ckpt = True
		if ser_micro_f1 > best_ser_micro_f1:
			best_ser_micro_f1 = ser_micro_f1
			should_save_ckpt = True
		if re_f1 > best_re_f1:
			best_re_f1 = re_f1
			should_save_ckpt = True
		if re_macro_f1 > best_re_macro_f1:
			best_re_macro_f1 = re_macro_f1
			should_save_ckpt = True

		if should_save_ckpt:
			ckpt_name = "epoch_{}_serF1_{:.04f}_{:.04f}_reF1_{:.04f}_{:.04f}.pdparams".format(
				epoch, ser_macro_f1, ser_micro_f1, re_f1, re_macro_f1)
			path_model = os.path.join(config['monitoring']['save_dir'], ckpt_name)
			P.save(model.state_dict(), path_model)



#start to eval
if __name__ == '__main__':

	main(config)
