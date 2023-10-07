#!/usr/bin/env python3
# -*- coding:utf-8 -*-


'''
@Project  kaiwu-fwk 
@File    sgame_client.py
@Author  : kaiwu
@Date    2022/5/23 21:05 

'''

import datetime
import socket
import os
import sys
import zmq
import struct
from multiprocessing import Process
import flatbuffers
from framework.common.config.config_control import CONFIG
from framework.common.utils.cmd_argparser import cmd_args_parse
from framework.common.logging.kaiwu_logger import KaiwuLogger
from framework.common.ipc.connection import Connection
from framework.server.python.aisrv.flatbuffer.kaiwu_msg import *
from framework.server.python.aisrv.flatbuffer.kaiwu_msg_helper import KaiwuMsgHelper
from app.sgame_1v1.env.protocl.proto_king.sgame_ai_server_pb2 import AIServerRequest, \
    AIServerResponse, AICommandInfo, KaiwuAIServerRequest, KaiwuAIServerResponse, SingleReq, PlayerInfo, GameInfo, \
    E_INIT, E_FRAME
from framework.common.utils.common_func import TimeIt

# 测试时, 可以直接修改后测试
CLIENT_VERSION =b'kaiwu_version_1.0'
PLAYER_NUM = 1

class SgameClient(Process):
    def __init__(self, client_id, client_version, player_num,zmq_port=35310):
        """
        用于给AIServer发送消息
        """
        super(SgameClient, self).__init__()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1000)
        sock.connect((CONFIG.aisvr_ip, int(CONFIG.aisvr_port)))
        self.conn = Connection(sock)
        self.client_id = b'sgame_client_1.0'
        self.client_version = client_version
        self.player_num = player_num

        self.zmq_ip = "127.0.0.1"
        self.zmq_port = [zmq_port]

        self.conns = []
        self.addrs = []
        self.game_over = False
        self.game_id = self.generate_game_id()

        # 记录发送的请求数目和时延
        self.send_count = 0
        self.send_recv_time_cost = 0.0

        #日志模块
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(f"/{CONFIG.svr_name}/battlesrv_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", CONFIG.svr_name)
        self.logger.info(f'battlesrv start at pid {pid}')
        self.logger.info("battlesrv 将使用zmq_port:{}".format(self.zmq_port[0]))
    
    def generate_game_id(self):
        dt = datetime.datetime.now()
        game_id = "gameid-" + dt.strftime("%Y%m%d-%H%M%S") + "-{}".format(self.client_id)
        return game_id

    def send_init_req(self, frame_no, msg: KaiwuAIServerRequest):
        """
        发送第一帧的消息
        :return:
        """
        builder = flatbuffers.Builder(0)
        init_req = KaiwuMsgHelper.encode_init_req(builder, self.client_id, '1.0', msg)
        req = KaiwuMsgHelper.encode_request(builder, frame_no, ReqMsg.ReqMsg.init_req, init_req)
        builder.Finish(req)
        req_msg = builder.Output()
        self.conn.send_msg(req_msg)
        self.logger.info('battlesrv send init req success')

        #获得rsp
        rsp_msg = self.conn.recv_msg()
        rsp = Response.Response.GetRootAsResponse(rsp_msg, 0)
        seq_no, _, init_rsp = KaiwuMsgHelper.decode_response(rsp)
        self.logger.info('battlesrv rec init req success')

    def recv_msg(self):
        rsp_msg = self.conn.recv_msg()
        rsp = KaiwuAIServerResponse()
        rsp.ParseFromString(rsp_msg)
        return rsp

    def prepare(self):
        #游戏开始前的准备

        # ep start req
        builder = flatbuffers.Builder(0)
        ep_start_req = KaiwuMsgHelper.encode_ep_start_req(builder, self.client_id, 0, b'')
        req = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.ep_start_req, ep_start_req)
        builder.Finish(req)
        req_msg = builder.Output()
        self.conn.send_msg(req_msg)
        self.logger.info("battlesrv Ep start")

        rsp_msg = self.conn.recv_msg() #rsp_msg其实并未用到
        for agent_id in range(self.player_num):
            # agent start req
            builder = flatbuffers.Builder(0)
            agent_start_req = KaiwuMsgHelper.encode_agent_start_req(builder, self.client_id, 0, agent_id, b'')
            req = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.agent_start_req, agent_start_req)
            builder.Finish(req)
            req_msg = builder.Output()
            self.conn.send_msg(req_msg)
            rsp_msg = self.conn.recv_msg()
            self.logger.info(f'battlesrv Agent {agent_id} start')

        self.logger.info("battlesrv Prepare OK")

    #发送每一帧的消息
    def send_update_req(self, frame_no, datas):
        #datas是一个数组，每一个元素是KaiwuAIServerRequest的字节码
        builder = flatbuffers.Builder(0)
        send_data = dict()
        for i in range(self.player_num):
            send_data[i] = [datas[i]]
        update_req = KaiwuMsgHelper.encode_update_req(builder, self.client_id, 0,
                                                      send_data)
        req = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.update_req, update_req)
        builder.Finish(req)
        req_msg = builder.Output()
        self.conn.send_msg(req_msg)
    
    def on_init(self):
        req = KaiwuAIServerRequest()
        req.player_num = self.player_num
        req.game_id =self.game_id
        player_info_list = []
        for i in range(self.player_num):
            player = PlayerInfo()
            player.hero_id = 123
            player_info_list.append(player)
        game_info = GameInfo()
        game_info.player_info.extend(player_info_list)
        req.game_info.CopyFrom(game_info)
        req.msg_type = E_INIT

        # 发送Init请求
        self.send_init_req(0, req.SerializeToString())
 
        # 准备
        self.prepare()

    def start_game(self):
        self._launch_zmq_server()

    def _launch_zmq_server(self):
        """
        启动zmq_server用来接收simulor的消息
        :return:
        """
        for i in range(self.player_num):
            context = zmq.Context()
            sockfd = context.socket(zmq.REP)
            addr = "tcp://{}:{}".format(self.zmq_ip, self.zmq_port[i])
            self.logger.info(f"battlesrv zmq addr {addr}")
            sockfd.bind(addr)
            self.conns.append(sockfd)
            self.addrs.append(addr)
    
    def _recv_msg_from_game(self, id):
        header = self.conns[id].recv()
        length = struct.unpack('>I', header[:4])[0]
        req_type = struct.unpack('I', header[4:8])[0]
        seq_no = struct.unpack('I', header[8:12])[0]
        pid = struct.unpack('I', header[12:16])[0]
        length -= 12
        obs = header[16:]

        return length, req_type, seq_no, obs, pid
    
    def recv_all_from_game(self):
        pbs = []
        # 假设只有一个send 会怎么样 -> 另一个会超时
        for id in range(self.player_num):
            rmsg = self._recv_msg_from_game(id)
            length, req_type, seq_no, obs, pid = rmsg
            req_pb = AIServerRequest()
            req_pb.ParseFromString(obs)
            if self.player_num == 2:
             #为了获得两个PB 直接发送
                rsp = self._get_rsp_pb(id)
                smsg = rsp.SerializeToString()
                msg = self._consturct_msg(smsg)
                self.conns[id].send(msg)

            self.frame_no = req_pb.frame_no

            single_req = SingleReq()
            single_req.ai_req.CopyFrom(req_pb)
            single_req.req_type = req_type
            single_req.seq_no = seq_no
            single_req.pid = pid
            pbs.append(single_req)

            if req_pb.gameover:
                self.game_over  =True
                self.logger.info("battlesrv really gameover!!!")
        return pbs

    def _consturct_msg(self, msg, ):
        total_len = 4 + len(msg)
        header = struct.pack('I', total_len)
        msg = header + msg
        return msg

    def on_update(self):
        """
        从aisrv获取数据响应返回给gamecore
        从gamecore获取数据请求发送给aisrv
        """
        pbs = self.recv_all_from_game()

        # 有两个pbs，这里做一个hack的方法
        data = KaiwuAIServerRequest()
        data.msg_type = E_FRAME
        data.player_num = self.player_num
        data.game_id = self.game_id
        for pb in pbs:
            data.req_list.append(pb)
        
        #这里改成了走UpdateReq分支
        datas = []
        for pb in pbs:
            datas.append(pb.SerializeToString())
        self.send_update_req(self.frame_no, datas)

        kaiwu_rsp:KaiwuAIServerResponse = self.recv_msg()
        if self.player_num == 1:
            rsp = kaiwu_rsp.rsp_pbs[0]
            smsg = rsp.SerializeToString()
            msg = self._consturct_msg(smsg)
            self.conns[0].send(msg)

        if self.game_over:
            self.logger.info(f"battlesrv 结束一局游戏, send/recv reqcount is {self.send_count}, avg time cost is {(self.send_recv_time_cost / self.send_count)*1000} ms")
            self.logger.info("总共帧数{}".format(self.frame_no))
            sys.exit(0)
    
    '''
    主函数
    '''
    def run(self):

        # 初始化
        self.on_init()

        # 开始游戏
        self.start_game()
        self.logger.info(f'battlesrv start game')

        while True:
            with TimeIt() as ti:
                self.on_update()

            self.send_count += 1
            self.send_recv_time_cost += ti.interval
            if self.send_count % 1000 == 0:
                self.logger.info('battlesrv 处理一帧, 花费时间 {} ms'.format(ti.interval*1000))

def proc_flags(configue_file):
    CONFIG.set_configue_file(configue_file)
    CONFIG.parse_client_configue()

def client_run(zmq_port, id, configure_file):
    os.chdir(CONFIG.project_root)

    # 解析参数
    proc_flags(configure_file)
    sgame_client = SgameClient(str(id), CLIENT_VERSION, PLAYER_NUM, zmq_port)
    return sgame_client

'''
启动命令样例: python client.py --conf=/data/projects/kaiwu-fwk/conf/client.ini
'''
if __name__ == '__main__':

    os.chdir(CONFIG.project_root)

    # 步骤1, 获取命令行参数
    args = cmd_args_parse("client")

    # 步骤2, 解析参数
    proc_flags(args.conf)

    # 步骤3, 启动client
    proc_num = int(CONFIG.gym_client_num)
    proc_list = []
    for i in range(proc_num):
        sgame_client = SgameClient(str(i), CLIENT_VERSION, PLAYER_NUM)

        sgame_client.start()
        proc_list.append(sgame_client)

    for i in range(proc_num):
        proc_list[i].join()
