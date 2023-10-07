#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# 模块介绍：本模块包含了样本处理的相关过程，样本处理的相关改动可以在这里实现

'''
@Project :kaiwu-fwk 
@File    :sgame_sample_processor.py
@Author  :kaiwu
@Date    :2022/6/15 20:57 

'''
import numpy as np
import collections
from app.sgame_1v1.env.protocl.proto_king.sgame_ai_server_pb2 import KaiwuAIServerRequest
from framework.common.utils.singleton import Singleton
from app.sgame_1v1.env.sample_processor.sgame_expr import SgameExpr as RLDataInfo
from app.sgame_1v1.common.configs.config import Config, ModelConfig
from framework.interface.sample_processor import SampleProcessor
from framework.common.config.config_control import CONFIG

IS_CHECK = Config.IS_CHECK
ACTION_DIM = Config.ACTION_DIM
INPUT_DIM = Config.INPUT_DIM

"""样本处理相关类"""
@Singleton
class SgameSampleProcessor(SampleProcessor):

    def __init__(self):

        #self.simu_ctx = simu_ctx

        self._data_shapes = ModelConfig.data_shapes
        self._LSTM_FRAME = ModelConfig.LSTM_TIME_STEPS

        # 由于aisrv生产样本时需要must_need_sample_info
        self.must_need_sample_info = None

        # load config from config file
        self.gamma = Config.GAMMA
        self.lamda = Config.LAMDA
        
        self.agent_policy= []

        # reward_sum
        self.log_rewsum = 0
    
    '''
    框架提供了日志接口, 框架使用者直接使用即可
    '''
    def set_logger(self, logger):
        self.logger = logger

    '''
    sample manager init 处理
    '''
    def on_init(self, player_num, game_id):
        self.game_id = game_id
        self.m_task_id, self.m_task_uuid = 0, "default_task_uuid"
        self.num_agents = player_num
        self.rl_data_map = [collections.OrderedDict() for _ in range(self.num_agents)]
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

        self.log_rewsum = 0

        self.logger.info(f"sample sample on_init success, game_id {self.game_id}, num_agents {self.num_agents}")
    
    def should_train(self):
        return True

    def gen_expr(self, must_need_sample_info, network_sample_info):
        """
        生成一个样本
        Args:
            must_need_sample_info: 由run_handler的on_update函数返回
            network_sample_info: Actor预测会返回action和网络参数, network_sample_info为样本需要的网络参数
        Returns:

        """
        self.must_need_sample_info = must_need_sample_info

        for i in range(self.num_agents):
            state_dict = must_need_sample_info[i]
            network_info = network_sample_info[i]

            is_train = False
            #req_pb = state_dict['req_pb']
            #frame_state = req_pb.frame_state
            #hero_camp = state_dict["hero_camp"]
            #for hero in frame_state.hero_states:
            #    if hero.actor_state.camp == hero_camp:
            #        is_train = True if hero.actor_state.hp > 0 else False
            is_train = state_dict['is_train']
            frame_no = state_dict["frame_no"]
            feature_vec, reward, sub_action_mask = state_dict['observation'], state_dict['reward'], state_dict['sub_action_mask']
            legal_action, action, value, prob, lstm_cell, lstm_hidden = network_info
            done = False
            keys = ("frame_no", "vec_feature", "legal_action", "action", "reward", "value", "prob", "sub_action",
                    "lstm_cell", "lstm_hidden", "done", "is_train")
            values = (frame_no, feature_vec, legal_action, action, reward[-1], value, prob, sub_action_mask,
                      lstm_cell, lstm_hidden, done, is_train)
            sample = dict(zip(keys, values))

            # TODO:只有最新的Model，才能产生Sample
            self.save_sample(**sample, agent_id=i, game_id=self.game_id, uuid=None)

        # self.logger.debug("sample gen_expr success")


    def proc_exprs(self, del_last=False):
        """
        生成一个Episode的全量样本
        Returns: train_data_all
        """
        total_frame_cnt = len(self.must_need_sample_info)
        

        #异常情况需要删除最后保存的样本才能保证样本的正确性
        if del_last:
            for i in range(self.num_agents):
                if len(list(self.rl_data_map[i].keys())) > 0:
                    last_key = list(self.rl_data_map[i].keys())[-1]
                    self.rl_data_map[i].pop(last_key)


        else:
            # TODO：只有最新模型才能生成样本
            for i in range(self.num_agents):
                state_dict = self.must_need_sample_info
                if state_dict[i]['reward'] != None:
                    if type(state_dict[i]['reward']) == tuple:
                        # if reward is a vec
                        self.save_last_sample(agent_id=i, reward=state_dict[i]['reward'][-1])
                    else:
                        # if reward is a float number
                        self.save_last_sample(agent_id=i, reward=state_dict[i]['reward'])

        train_data = self.send_samples()
        return_rew = self.log_rewsum

        # 对train_data进行压平处理
        train_data_all = []
        for agent_data in train_data:
            # agent_data:list[(frame_no,vec)]
            for sample in agent_data:
                train_data_all.append({
                    # 发送样本时, 强制转换成float16
                    'input_datas': np.array(sample[1], dtype=np.float16)
                })
        train_frame_cnt = len(train_data)
        drop_frame_cnt = total_frame_cnt - train_frame_cnt
        self.logger.info(f'sample train_frame_cnt {train_frame_cnt},  drop_frame_cnt {drop_frame_cnt}, reward {return_rew}')
        return train_data_all, train_frame_cnt, return_rew

    def reset(self, num_agents, game_id):
        self.game_id = game_id
        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(self.num_agents)]
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

        self.log_rewsum = 0

    def save_sample(self, frame_no,
                    vec_feature, legal_action, action, reward, value, prob, sub_action,
                    lstm_cell, lstm_hidden,
                    done, agent_id, is_train=True,
                    game_id=None, uuid=None):
        """
        samples must saved by frame_no order
        """
        reward = self._clip_reward(reward)
        #只需要上报id=0的reward sum
        if agent_id==0:
            self.log_rewsum += reward

        rl_data_info = RLDataInfo()

        value = value.flatten()[0]
        lstm_cell = lstm_cell.flatten()
        lstm_hidden = lstm_hidden.flatten()

        # update last frame's next_value
        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = value
            last_rl_data_info.reward = reward

        # save current sample
        rl_data_info.frame_no = frame_no

        rl_data_info.feature = vec_feature.reshape([-1])
        rl_data_info.legal_action = legal_action.reshape([-1])
        rl_data_info.reward = 0
        rl_data_info.value = value
        # rl_data_info.done = done     
        rl_data_info.lstm_info = np.concatenate([lstm_cell, lstm_hidden]).reshape([-1])
        # np: (12 + 16 + 16 + 16 + 14)
        rl_data_info.prob = prob
        # np: (6)
        # rl_data_info.action = 0 if action < 0 else action
        rl_data_info.action = action
        # np: (6)
        rl_data_info.sub_action = sub_action[action[0]]
        rl_data_info.is_train = False if action[0] < 0 else is_train

        self.rl_data_map[agent_id][frame_no] = rl_data_info
        
    def save_last_sample(self, reward, agent_id):
        self.logger.info("sample save last sample")
        if len(self.rl_data_map[agent_id]) > 0:
            # TODO: is_action_executed, last_gamecore_act
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = 0
            last_rl_data_info.reward = reward

    def send_samples(self):
        self._calc_reward()
        self._format_data()

        return self._send_game_data()

    def _calc_reward(self):
        """
        Calculate cumulated reward and advantage with GAE.
        reward_sum: used for value loss
        advantage: used for policy loss
        V(s) here is a approximation of target network
        """
        for i in range(self.num_agents):
            reversed_keys = list(self.rl_data_map[i].keys())
            reversed_keys.reverse()
            gae, last_gae = 0., 0.
            for j in reversed_keys:
                rl_info = self.rl_data_map[i][j]
                delta = -rl_info.value + rl_info.reward + self.gamma * rl_info.next_value
                gae = gae * self.gamma * self.lamda + delta
                rl_info.advantage = gae
                rl_info.reward_sum = gae + rl_info.value

    def _reshape_lstm_batch_sample(self, sample_batch, sample_lstm):
        sample = np.zeros([np.prod(sample_batch.shape) + np.prod(sample_lstm.shape)])
        sample_one_size = sample_batch.shape[1]
        idx, s_idx = 0, 0

        sample[-sample_lstm.shape[0]:] = sample_lstm
        for split_shape in self._data_shapes[:-2]:
            one_shape = split_shape[0] // self._LSTM_FRAME
            sample[s_idx: s_idx + split_shape[0]] = sample_batch[:, idx: idx + one_shape].reshape([-1])
            idx += one_shape
            s_idx += split_shape[0]
        return sample

    def _format_data(self):
        sample_one_size = np.sum(self._data_shapes[:-2]) // self._LSTM_FRAME
        sample_lstm_size = np.sum(self._data_shapes[-2:])
        sample_batch = np.zeros([self._LSTM_FRAME, sample_one_size])
        sample_lstm = np.zeros([sample_lstm_size])
        first_frame_no = -1

        for i in range(self.num_agents):
            cnt = 0
            for j in self.rl_data_map[i]:
                rl_info = self.rl_data_map[i][j]

                if cnt == 0:
                    # lstm cell & hidden
                    first_frame_no = rl_info.frame_no
                    sample_lstm = rl_info.lstm_info

                # serilize one frames
                idx, dlen = 0, 0
                # vec_data
                dlen = rl_info.feature.shape[0]
                sample_batch[cnt, idx:idx + dlen] = rl_info.feature
                idx += dlen

                # legal_action
                dlen = rl_info.legal_action.shape[0]
                sample_batch[cnt, idx:idx + dlen] = rl_info.legal_action
                idx += dlen

                # reward_sum & advantage
                sample_batch[cnt, idx] = rl_info.reward_sum
                idx += 1
                sample_batch[cnt, idx] = rl_info.advantage
                idx += 1

                # labels
                dlen = 6
                sample_batch[cnt, idx:idx + dlen] = rl_info.action
                idx += dlen

                # probs (neg log pi->prob)
                for p in rl_info.prob:
                    dlen = len(p)
                    # p = np.exp(-nlp)
                    # p = p / np.sum(p)
                    sample_batch[cnt, idx:idx + dlen] = p
                    idx += dlen

                # sub_action
                dlen = 6
                sample_batch[cnt, idx:idx + dlen] = rl_info.sub_action
                idx += dlen

                # is_train
                sample_batch[cnt, idx] = rl_info.is_train
                idx += 1

                assert idx == sample_one_size, "Sample check failed, {}/{}".format(idx, sample_one_size)

                cnt += 1
                if cnt == self._LSTM_FRAME:
                    cnt = 0
                    sample = self._reshape_lstm_batch_sample(sample_batch, sample_lstm)
                    self.m_replay_buffer[i].append((first_frame_no, sample))
                    # self.logger.debug(f'sample first_frame_no {first_frame_no} add sample success')
                
            #尾部样本不丢弃
            if cnt>0:
                a_ins=len(list(self.rl_data_map[i].keys()))-self._LSTM_FRAME
                e_ins=len(list(self.rl_data_map[i].keys()))
                cnt=0
                if a_ins >= 0:
                    for k in range(a_ins,e_ins):
                        rl_info = self.rl_data_map[i][list(self.rl_data_map[i].keys())[k]]

                        if cnt == 0:
                            # lstm cell & hidden
                            first_frame_no = rl_info.frame_no
                            sample_lstm = rl_info.lstm_info

                        # serilize one frames
                        idx, dlen = 0, 0
                        # vec_data
                        dlen = rl_info.feature.shape[0]
                        sample_batch[cnt, idx:idx + dlen] = rl_info.feature
                        idx += dlen

                        # legal_action
                        dlen = rl_info.legal_action.shape[0]
                        sample_batch[cnt, idx:idx + dlen] = rl_info.legal_action
                        idx += dlen

                        # reward_sum & advantage
                        sample_batch[cnt, idx] = rl_info.reward_sum
                        idx += 1
                        sample_batch[cnt, idx] = rl_info.advantage
                        idx += 1

                        # labels
                        dlen = 6
                        sample_batch[cnt, idx:idx + dlen] = rl_info.action
                        idx += dlen

                        # probs (neg log pi->prob)
                        for p in rl_info.prob:
                            dlen = len(p)
                            # p = np.exp(-nlp)
                            # p = p / np.sum(p)
                            sample_batch[cnt, idx:idx + dlen] = p
                            idx += dlen

                        # sub_action
                        dlen = 6
                        sample_batch[cnt, idx:idx + dlen] = rl_info.sub_action
                        idx += dlen

                        # is_train
                        sample_batch[cnt, idx] = rl_info.is_train
                        idx += 1

                        assert idx == sample_one_size, "Sample check failed, {}/{}".format(idx, sample_one_size)

                        cnt += 1
                        if cnt == self._LSTM_FRAME:
                            cnt = 0
                            sample = self._reshape_lstm_batch_sample(sample_batch, sample_lstm)
                            self.m_replay_buffer[i].append((first_frame_no, sample))
                        

    def _clip_reward(self, reward, max=100, min=-100):
        if reward > max:
            reward = max
        elif reward < min:
            reward = min
        return reward

    # send game info like: ((size, data))*5:
    # [task_id: int, task_uuid: str, game_id: str, frame_no: int, real_data: data in str]

    def _send_game_data(self):
        all_samples = []
        
        # 不保存self-play时用旧模型训练的样本
        for i in range(self.num_agents):
            if not (CONFIG.self_play and self.agent_policy[i] == CONFIG.self_play_old_policy):
                all_samples.append(self.m_replay_buffer[i])
        
        # self.logger.info(f"sample send game data {all_samples}")
        return all_samples
