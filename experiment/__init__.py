# -*- coding:utf-8 -*-
# email:
# create: 2021/6/8

from .custom_diffusion_experiment import CustomDiffusionExperiment


def get_experiment_name(name):
    name_split = name.split("_")
    trainer_name = "".join([tmp_name[0].upper() + tmp_name[1:] for tmp_name in name_split])
    return "{}Experiment".format(trainer_name)