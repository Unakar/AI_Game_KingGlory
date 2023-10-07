#!/usr/bin/env python3
# -*- coding:utf-8 -*-


'''
@Project : kaiwu-fwk 
@File    : sgame_action.py
@Author  : kaiwu
@Date    : 2022/6/13 15:31 

'''
import numpy as np
from framework.interface.array_spec import ArraySpec
from framework.common.algorithms.distribution import CategoricalDist
from framework.interface.action import Action, ActionSpec

# 模块介绍：本模块描述了action space信息，主要用于actor上使用

class SgameAction(Action):
    def __init__(self, a):
        self.a = a

    def get_action(self):
        return {'a': self.a}

    @staticmethod
    def action_space():
        action_space = 50
        return {'a': ActionSpec(ArraySpec((action_space, ), np.int32), pdclass=CategoricalDist)}

    def __str__(self):
        return str(self.a)
