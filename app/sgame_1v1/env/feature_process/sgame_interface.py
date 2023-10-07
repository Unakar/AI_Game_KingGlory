#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :kaiwu-fwk 
@File    :sgame_interface.py
@Author  :kaiwu
@Date    :2022/6/15 20:57 

'''

import os
import app.sgame_1v1.env.feature_process.interface as interface

'''
特征处理和返回的接口，具体由C++实现，interface.so
C++代码地址: https://git.woa.com/king-kaiwu/ai/dockerfile_gamecore_pb_process
'''
class SgameInterface:
    def __init__(self, configure_file, logger) -> None:

        self.lib_processor = interface.Interface()

        os.chdir('/data/projects/kaiwu-fwk/app/sgame_1v1/env/feature_process/')
        self.lib_processor.Init(configure_file)
        self.lib_processor.ReSet(False)

        logger.info("c++ interface Init success")
    
    def feature_process(self, length, req_type, seq_no, obs, id):
        return self.lib_processor.FeatureProcess(length, req_type, seq_no, obs, id)

    def result_proces(self, format_actions, id):
        return self.lib_processor.ResultProcess(format_actions, id)