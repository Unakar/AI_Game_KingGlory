from multiprocessing import Process
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from framework.common.utils.common_func import stop_process_by_name, python_exec_shell
from tools.learner_step_check import countCkpt
import framework.server.python.learner.learner as learner
import framework.server.python.actor.actor as actor
import framework.server.python.aisrv.aisrv as aisrv
import time
import requests as req
import click
import os

# 训练


@click.command()
@click.option('--battlesvr_addr', default='127.0.0.1:12345', help='batch_size参数，非必须，有默认值')
def train(battlesvr_addr='127.0.0.1:12345'):
    '''
    '''

    cleanProcesses()

    # 启动训练相关进程
    procs = []
    procs.append(Process(target=learner.main, name="learner"))
    procs.append(Process(target=actor.main, name="actor"))
    procs.append(Process(target=aisrv.main, name="aisrv"))
    procs.append(Process(target=python_exec_shell, args=(
        'sh tools/start_modelpool.sh learner',), name='modelpool'))

    for proc in procs:
        proc.start()
        time.sleep(10)


    # 计算已有的checkpoint数量
    oldCkpt = countCkpt()

    # 启动对战
    stopBattle(battlesvr_addr)
    startBattle(battlesvr_addr)

    # 监听进程是否退出
    code = 0
    while True:
        if code > 0:
            break

        newCkpt = countCkpt()
        # 有新的checkpoint产出即退出
        if newCkpt - oldCkpt > 0:
            print("test successful")
            break

        time.sleep(2)
        for proc in procs:
            if not proc.is_alive():
                print(f'{proc.name} is not alive, please check error log')
                code = 1

    cleanProcesses()

    print(f"will exit: {code}")
    exit(code)


def cleanProcesses():
    """
    清理残留进程
    """

    # 清理除了当前进程外所有名为 train_test.py 的进程
    current_pid = os.getpid()
    script = "ps aux | grep train_test.py | grep -v %d |grep -v debugpy| awk '{print $2}' | xargs -r kill" % current_pid
    python_exec_shell(script)

    # 清理相关进程
    stop_process_by_name(KaiwuDRLDefine.SERVER_ACTOR)
    stop_process_by_name(KaiwuDRLDefine.SERVER_LEARNER)
    time.sleep(10)
    stop_process_by_name(KaiwuDRLDefine.SERVER_MODELPOOL)
    stop_process_by_name(KaiwuDRLDefine.SERVER_MODELPOOL_PROXY)


def startBattle(battlesvr_addr):
    rsp = req.post(
        f'http://{battlesvr_addr}/kaiwu_drl.BattleSvr/Start', json={"max_battle": 1})
    if rsp.status_code > 300:
        raise Exception("start battle fail")


def stopBattle(battlesvr_addr):
    rsp = req.post(
        f'http://{battlesvr_addr}/kaiwu_drl.BattleSvr/Stop', json={})
    if rsp.status_code > 300:
        raise Exception("stop battle fail")


if __name__ == '__main__':
    train()
