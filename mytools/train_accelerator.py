# -*- coding:utf-8 -*-
# email:
# create: 2021/6/10
import os
import sys
import argparse
import setproctitle

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT_PATH)
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# accelerate launch --config_file path_to_config.yaml path_to_script.py
import experiment

from base.common_util import init_experiment_config
from experiment import get_experiment_name


def init_args():
    parser = argparse.ArgumentParser(description='trainer args')
    parser.add_argument(
        '--config_file',
        default='config/custom_diffusion/dream_booth_base.yaml',
        type=str,
    )
    parser.add_argument(
        '--experiment_name',
        default='CustomDiffusion',
        type=str,
    )
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
    )
    args = parser.parse_args()
    os.environ['WORKSPACE'] = args.experiment_name
    return args


def main(args):
    config = init_experiment_config(args.config_file, args.experiment_name)
    config.update({'phase': args.phase})
    experiment_instance = getattr(experiment, get_experiment_name(args.experiment_name))(config)
    if args.phase == 'train':
        experiment_instance.train()
    elif args.phase == 'predict':
        experiment_instance.predict()
    elif args.phase == 'evaluate':
        experiment_instance.evaluate()


if __name__ == '__main__':
    args = init_args()
    setproctitle.setproctitle("{} task for {}".format(args.experiment_name, args.config_file.split('/')[-1]))
    main(args)
