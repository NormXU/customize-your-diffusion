# -*- coding:utf-8 -*-
# email:
# create: 2021/6/8
from mydatasets.base_datasets import BaseDataset, BaseImgDataset
from .custom_diffusion_dataset import CustomDiffusionDataset, CustomDiffusionCollectFn


def get_dataset(dataset_args):
    dataset_type = dataset_args.get("type")
    dataset = eval(dataset_type)(**dataset_args)
    return dataset
