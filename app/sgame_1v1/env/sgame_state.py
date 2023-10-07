#!/usr/bin/env python3
# -*- coding:utf-8 -*-


'''
@Project : kaiwu-fwk 
@File    : sgame_state.py
@Author  : kaiwu
@Date    : 2022/6/13 11:31 

'''
import numpy as np
from framework.interface.array_spec import ArraySpec
from framework.interface.state import State

# 模块介绍：本模块描述了state space信息，主要用于actor上使用

class SgameState(State):
    def __init__(self, value):
        """
        Args:
            value: 由run_handler构造本类, 为on_update函数的一个返回值(当需要预测时)
        """
        self.value = value

    def get_state(self):
        """
        根据构造函数中传入的value,构造返回一个dict
        dict会传给Actor进行预测
        """
        observation = np.array(self.value["observation"],dtype=np.float64)
        legal_action = np.array(self.value['legal_action'],dtype=np.float64)
        sub_action_mask = self.value['sub_action_mask']
        sub_action_mask = np.stack(list(sub_action_mask.values()), axis=0)
        lstm_hidden = self.value['lstm_hidden']
        lstm_cell = self.value['lstm_cell']
        return {
                'observation': observation,
                'legal_action':legal_action,
                'sub_action_mask':sub_action_mask,
                'lstm_hidden':lstm_hidden,
                'lstm_cell':lstm_cell
                }

    @staticmethod
    def state_space():
        """
        规定state中每个变量的shape, 必须为numpy数组
        Returns:
        """
        observation_shape = (725,)
        legal_action_shape = (172,)
        sub_action_mask_shape =(12,6)
        lstm_hidden_shape = (512,)
        lstm_cell_shape = (512,)
        return {
                'observation':  ArraySpec(observation_shape, np.float64),
                'legal_action': ArraySpec(legal_action_shape, np.float64),
                'sub_action_mask': ArraySpec(sub_action_mask_shape,np.float64),
                'lstm_hidden':ArraySpec(lstm_hidden_shape,np.float64),
                'lstm_cell': ArraySpec(lstm_cell_shape,np.float64)
                }

    def __str__(self):
        return str(self.value)
 
