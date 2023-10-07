#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# 模块介绍：本模块描述了reward scale的信息

'''
@Project  kaiwu-fwk 
@File    sgame_reward_shaper.py
@Author  : kaiwu
@Date    2022/5/23 21:05 

'''

from framework.common.config.config_control import CONFIG
from framework.interface.reward_shaper import RewardShaper

class SgameRewardShaper(RewardShaper):
    def __init__(self, simu_ctx, agent_ctx):
        super().__init__(simu_ctx, agent_ctx)

        self.t = 0

    def should_train(self, exprs):
        self.t += 1
        done = exprs[-1].done

        return done or self.t == 128

    def assign_rewards(self, exprs):
        for expr in exprs:
            expr.reward.add_in_reward(0.0)
        self.t = 0
