""" eval_infer.py """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import sys
import json
import logging
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
from external.evaler import Evaler
from external.trainer import Trainer

np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# sysconf
# base
parser = argparse.ArgumentParser('launch for eval')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--task_type', type=str, required=True)
parser.add_argument('--weights_path', type=str, required=True)

args = parser.parse_args()
print_arguments(args)
config = json.loads(open(args.config_file).read())

ALL_MODULES = ['labeling_segment', 'labeling_token', 'linking']
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

	assert weights_path.endswith('.pdparams') and \
			os.path.isfile(weights_path), \
			'the weights_path %s is not existed!' % weights_path

	config['init_model'] = weights_path
	train_config['dataset']['max_seqlen'] = \
			model_config['embedding']['max_seqlen']

	place = P.set_device('cpu')

	train_dataset = Dataset(
			train_config['dataset'],
			train_config['feed_names'],
			False
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
	eval_classes = build_metric(train_config['metric'])

	#start
#     logging.info('eval start...')
	trainer = Trainer(config, model, train_loader)
	evaler = Evaler(config, model, eval_loader, eval_classes)
	print("Strat training........................")
	trainer.train()

	print("Strat evaluation............")
	evaler.run()
	logging.info('eval end...')

#start to eval
if __name__ == '__main__':

	main(config)
