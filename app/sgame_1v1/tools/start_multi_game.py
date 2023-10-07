#!/usr/bin/python3
# -*- coding: UTF-8 -*-


'''
@Project : kaiwu-fwk 
@File    : start_multi_game.py
@Author  : kaiwu
@Date    : 2022/6/29 9:54 

'''
import datetime
import os
import random
import signal
import subprocess
import sys
import atexit

def clear_log_and_stat():
    print("清除多余日志")
    command = "cd {};rm gameid-*.stat;rm *.abs".format(game_path)
    subprocess.Popen([command], preexec_fn=os.setpgrp, shell=True)
    print("kill可能未结束的游戏")
    print(pid_list)
    # time.sleep(10)
    for pid in pid_list:
        try:
            os.killpg(pid, signal.SIGKILL)
            print("kill {}".format(pid))
        except Exception as e:
            print(e)
            continue
    print("结束清理")


pid_list = []
def f():
    print('结束')
    clear_log_and_stat()


atexit.register(f)

from app.sgame_1v1.tools.sgame_client import client_run

BASE_PORT = 35310
game_path = os.getenv("GAME_PATH")
if game_path == None:
    print("请设置环境变量GAME_PATH,为game文件夹的目录, 实例export GAME_PATH=/data/projects/sgame/game/game/")
    exit(-1)
    # game_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "game")


def read_conf_file():
    with open(os.path.join(game_path, "sgame_simulator.conf"), "r") as f:
        datas = list(map(str.strip, f.readlines()))
    return datas


def format_game_cmd(game_id, conf_file, id):
    command = "cd {};".format(game_path) + \
              "LD_LIBRARY_PATH=\"./:./core_assets:${{LD_LIBRARY_PATH}}\" " \
              "nohup stdbuf -oL ./sgame_simulator_remote_zmq \'{}\' \'./{}\' >> ./game_{}.log 2>&1".format(
                  game_id, conf_file, id)
    return command


def get_client_cmd(port, id):
    conda_path = os.popen("echo `which conda`").readlines()[0].replace("conda", "activate")
    command = "source {} kaiwu && nohup python sgame_client.py {}  >> ./client_{}.log 2>&1".format(conda_path, port, id)
    return command


def generate_game_id(game_seq):
    dt = datetime.datetime.now()
    game_id = "gameid-" + dt.strftime("%Y%m%d-%H%M%S") + "-{}".format(game_seq)
    return game_id


def generate_conf_file(base_conf, port):
    file_name = "sgame_simulator_{}.conf".format(port)
    with open(os.path.join(game_path, file_name), "w") as f:
        for i, line in enumerate(base_conf):
            if i == 2:
                line = str(line).replace(str(BASE_PORT), str(port))
            f.write(line + "\n")
    return file_name


def start_game(game_num, configure_file):
    base_conf = read_conf_file()
    proc_list = []
    offset = random.randint(50, 100)
    for i in range(game_num):
        print("启动第{}场游戏".format(i))

        port = BASE_PORT + offset + i
        file_name = generate_conf_file(base_conf, port)
        game_id = generate_game_id(i + 1)
        sgame_client = client_run(port, i, configure_file)
        sgame_client.start()
        proc_list.append(sgame_client)

        command = format_game_cmd(game_id, file_name, i)
        pid = subprocess.Popen([command], preexec_fn=os.setpgrp, shell=True)
        pid_list.append(pid.pid)

    for i in range(game_num):
        proc_list[i].join()
    
    clear_log_and_stat()

'''
进程启动实例: python start_multi_game.py 1 /data/projects/kaiwu-fwk/conf/client.ini
'''

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage:python start_multi_game.py 游戏数目 配置文件")
        exit()
    
    try:
        game_num = int(sys.argv[1])
        print("收到命令行参数，将产生{}局游戏".format(game_num))
    except Exception as e:
        print("未收到命令行参数，将产生一局游戏")
        game_num = 1
    
    for i in range(10000):
        print("开始第{}轮".format(i))
        start_game(game_num, sys.argv[2])
        pid_list.clear()
