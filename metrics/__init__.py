# -*- coding:utf-8 -*-
# email:
# create: 2021/7/1


def get_metric(metric_args):
    metric_type = metric_args['type']
    return eval(metric_type)(**metric_args)
