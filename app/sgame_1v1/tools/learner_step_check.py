#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project  kaiwu-fwk 
@File    sgame_client.py
@Author  : kaiwu
@Date    2022/5/23 21:05 

'''

import os

def countCkpt():
    '''
    查看learner机器上的checkpoint数量
    '''
    addr = '/data/ckpt/sgame_1v1_ppo/checkpoint'
    if not os.path.exists(addr):
        print(f'{addr} file not exists, please check!')
        return 0
    
    res = os.popen('tail -n 1 /data/ckpt/sgame_1v1_ppo/checkpoint').read()
    if len(res) == 0:
        return 0

    tmp = res.split('-')[-1]
    cnt = tmp.split('"')[0]

    return int(cnt)
