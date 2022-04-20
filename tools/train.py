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
from utils.log import setup_logger
from external.evaler import Evaler
from external.trainer import Trainer
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

ALL_MODULES = ['labeling_segment', 'labeling_token', 'linking', 'joint']
if args.task_type not in ALL_MODULES:
	raise ValueError('Not valid task_type %s in %s' % (args.task_type, str(ALL_MODULES)))

# choose trainer
TrainerClass = Trainer
if args.task_type == 'joint':
	TrainerClass = JointTrainer

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
	logging = setup_logger('logs',
						   os.path.join(config['monitoring']['save_dir'],
						   				config['monitoring']['log_file']))

	config['init_model'] = weights_path
	train_config['dataset']['max_seqlen'] = \
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
	eval_classes = build_metric(train_config['metric'])

	trainer = TrainerClass(config, model, train_loader, logging)
	evaler = Evaler(config, model, eval_loader, eval_classes)

	logging.info("Start training........................")
	trainer.train()

	logging.info("Start evaluation............")
	evaler.run()
	
	logging.info('eval end...')

#start to eval
if __name__ == '__main__':

	main(config)
