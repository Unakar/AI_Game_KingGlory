#! /bin/bash

# 启动sgame_client, 同时启动gamecore

if [ $# -ne 2 ];
then
    echo  -e "\033[31m useage: sh start_multi_game.sh run_times conf_file, such as: sh start_multi_game.sh 1 /data/projects/kaiwu-fwk/conf/client.ini \033[0m"
    exit -1
fi

run_times=$1
conf_file=$2

# 设置game_path
export GAME_PATH=/data/projects/sgame/game/game/

# 先kill掉已经有的进程
process_name=start_multi_game
process_num=`ps -ef | grep $process_name | grep -v grep | grep -v start_multi_game.sh | wc -l`
if [ $process_num -gt 0 ];
then
    ps -ef | grep $process_name | grep -v "grep" | awk '{print $2}' | xargs kill -9
fi

python3 start_multi_game.py $run_times $conf_file
if [ $? -ne 0 ];
then
    echo  -e "\033[32m sh start_multi_game.sh $run_times $conf_file, success \033[0m"
else
    echo  -e "\033[31m sh start_multi_game.sh $run_times $conf_file, failed \033[0m"
fi