#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# 模块介绍：本模块描述了样本信息，包括每个样本包含的信息以及默认值

'''
@Project :kaiwu-fwk 
@File    :sgame_expr.py
@Author  :kaiwu
@Date    :2022/6/15 20:57 

'''

class SgameExpr:
    def __init__(self):
        # 每个样本包含的信息以及默认值
        self.frame_no = -1
        self.feature = b""
        self.next_feature = b""
        self.reward = 0
        self.reward2 = 0
        self.reward_sum = 0
        self.reward_sum2 = 0
        self.action = 0
        self.action_list = []
        self.done = 0
        self.info = None
        self.value = 0
        self.value2 = 0
        # self.neg_log_pis = 0
        self.advantage = 0
        self.game_id = b""
        self.is_train = False
        self.is_game_over = 0
        self.task_uuid = b""
        self.next_Q_value = b""
        self.gamma_pow = 1

        self.prob = None
        self.sub_action = None
        self.next_value = 0
        self.next_value2 = 0
        self.lstm_info = None

    def __str__(self) -> str:
        return f'frame_no {self.frame_no}, feature {self.feature}, next_feature {self.next_feature}, reward {self.reward }, reward2 {self.reward2}, \
            reward_sum {self.reward_sum}, reward_sum2 {self.reward_sum2}, action {self.action}, action_list {self.action_list}, done {self.done}, \
                info {self.info}, value {self.value }, value2 {self.value2}, advantage {self.advantage}, game_id {self.game_id}, is_train {self.is_train}, \
                    is_game_over {self.is_game_over}, task_uuid {self.task_uuid}, next_Q_value {self.next_Q_value}, gamma_pow {self.gamma_pow}, \
                        prob {self.prob}, sub_action {self.sub_action }, next_value {self.next_value}, next_value2 {self.next_value2}, lstm_info {self.lstm_info}'