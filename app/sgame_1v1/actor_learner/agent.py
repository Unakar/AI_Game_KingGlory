#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# 模块介绍：本模块包含强化学习概念中agent的实现，用户自定义的agent可以在本模块中实现

'''
@Project :kaiwu-fwk 
@File    :agent.py
@Author  :kaiwu
@Date    :2022/6/15 20:57 

'''

import traceback
import numpy as np
import random
import h5py
from framework.common.checkpoint.model_pool_apis import ModelPoolAPIs
from app.sgame_1v1.actor_learner.local_predictor import LocalCkptPredictor as LocalPredictor
from app.sgame_1v1.actor_learner.utils import cvt_tensor_to_infer_input, cvt_tensor_to_infer_output
from app.sgame_1v1.common.configs.config import ModelConfig, Config


_g_check_point_prefix = "checkpoints_"
_g_rand_max = 10000
_g_model_update_ratio = 0.8

def cvt_infer_list_to_numpy_list(infer_list):
    data_list = [infer.data for infer in infer_list]
    return data_list

class RandomAgent():
    '''
        一个输出随机动作Agent的简单实现
        可以用作流程测试或者拓展为启发式AI实现
    '''
    def process(self, feature, legal_action):
        action = [random.randint(0, 2) - 1, random.randint(0, 2) - 1]
        value = [0.0]
        neg_log_pi = [0]
        return action, value, neg_log_pi

class Agent:
    '''
        强化学习概念中agent的实现，包含了模型和模型决策接口
    '''
    def __init__(self, model_cls, model_pool_addr, keep_latest=False, local_mode=False, dataset=None):
        self.model = model_cls()
        # 构造模型graph
        self.graph = self.model.build_infer_graph()
        # state->action的预测器
        self._predictor = LocalPredictor(self.graph)
        if local_mode:
            self._model_pool_api = None
        else:
            self._model_pool_api = ModelPoolAPIs(model_pool_addr)

        self.model_version = ""
        self.is_latest_model: bool = False
        self.keep_latest = keep_latest
        self.model_list = []

        # lstm_*用于lstm信息保存，给agent提供长序列决策的记忆功能
        self.lstm_unit_size = ModelConfig.LSTM_UNIT_SIZE

        self.lstm_hidden = None
        self.lstm_cell = None

        # self.agent_type = "common_ai"
        self.player_id = 0
        self.hero_camp = 0
        self.last_model_path = None

        # 输出决策的信息，包括决策维度和合法动作
        self.label_size_list = ModelConfig.LABEL_SIZE_LIST
        self.legal_action_size = ModelConfig.LEGAL_ACTION_SIZE_LIST

        # self.agent_type = "network"
        if self.keep_latest:
            self.agent_type = "network"
        else:
            self.agent_type = Config.ENEMY_TYPE
        
        # 是否载入现有数据集训练，默认不载入
        if dataset is None:
            self.save_h5_sample = False
            self.dataset_name = None
            self.dataset=None
        else:
            self.save_h5_sample = True
            self.dataset_name = dataset
            self.dataset = h5py.File(dataset, 'a')
        
        # 日志接口从框架侧提供
        self.logger = None
 
    def set_logger(self, logger):
        '''
        框架提供了日志接口, 框架使用者直接使用即可
        '''
        self.logger = logger

    def set_game_info(self, hero_camp, player_id):
        self.hero_camp = hero_camp
        self.player_id = player_id

    def reset(self, agent_type=None, model_path=None):
        # reset lstm input
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])

        if agent_type is not None:
            if self.keep_latest:
                self.agent_type = "network"
            else:
                self.agent_type = agent_type
        
        # 测试专用
        self.keep_latest = False
        self.agent_type = "random"

        # for test without model pool
        if Config.SINGLE_TEST:
            self.is_latest_model = True
            self._predictor._sess.run(self.model.init)
        else:
            if model_path is None:
                while True:
                    try:
                        if self.keep_latest:
                            self._get_latest_model()
                        else:
                            self._get_random_model()
                        
                        self.last_model_path = None
                        return
                    except Exception as e:
                        self.logger.error(f"sgame_learner agent get_model error, try again..., error is {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
            else:
                if model_path != self.last_model_path:
                    ret = self._predictor.load_model(model_path)
                    self.last_model_path = model_path
                else:
                    self.logger.info("sgame_learner model {} alreadly load last time, skip now!".format(model_path))

        if self.dataset is None:
            self.save_h5_sample = False
        else:
            self.save_h5_sample = True
            self.dataset.close()
            self.dataset = h5py.File(self.dataset_name, 'a')


    def _update_model_list(self):
        '''
        更新从learner同步的model文件列表
        '''
        import time
        model_key_list = []
        while len(model_key_list) == 0:
            model_key_list = self._model_pool_api.pull_keys()
            if model_key_list is None:
                self.logger.warning("sgame_learner agent No model in model_pool, wait for 1 sec...")
                time.sleep(1)
        self.model_list = model_key_list
    
    def load_last_new_model(self, models_path):
        '''
        加载从learner同步最新的model文件
        '''
        self._predictor.load_last_model(models_path)

    def _load_model(self, model_version):
        '''
        实现载入模型的方法，原则上不对外开放调用
        '''
        if model_version == self.model_version:
            return True
        model_path = self._model_pool_api.pull_model_path(model_version)
        model_path = "%s/checkpoint" % (model_path)
        self.logger.info("sgame_learner agent load model: {} in {}".format(model_version, model_path))
        ret = self._predictor.load_model(model_path)
        if ret:
            # if failed, do not update model_version
            self.model_version = model_version
        return ret
    
    def _get_random_model(self):
        '''
        在model list中载入随机模型
        '''
        if self.agent_type in ["common_ai", "random"]:
            self.is_latest_model = False
            self._predictor._sess.run(self.model.init)
            self.model_version = ""
            return True

        self._update_model_list()
        rand_float = float(random.uniform(0, _g_rand_max)) / float(_g_rand_max)
        if rand_float <= _g_model_update_ratio:
            midx = len(self.model_list)-1
            self.is_latest_model = True
        else:
            midx = int(random.random() * len(self.model_list))
            if midx == len(self.model_list):
                midx = len(self.model_list)-1
            self.is_latest_model = False
        return self._load_model(self.model_list[midx])

    def _get_latest_model(self):
        '''
        载入最新模型
        '''
        self._update_model_list()
        self.is_latest_model = True
        return self._load_model(self.model_list[-1])

    def process(self, state_dict, battle=False):
        '''
        主函数，包括预测和样本处理
        '''
        # todo add legal action process, should moved from model _legal_soft_max()
        # print("legal_action",state_dict["legal_action"])
        feature_vec, legal_action = state_dict["observation"], state_dict["legal_action"]
        pred_ret = self._predict_process(feature_vec, legal_action)
        # prob, value, action
        prob, value, action, d_action = pred_ret
        if battle:
            return d_action
        
        return action, d_action, self._sample_process(state_dict, pred_ret)

    def _update_legal_action(self, original_la, actions):
        '''
        更新当前的legal_action, 例如上一帧释放1技能后，下一帧在CD中则legal更新为false
        '''
        target_size = ModelConfig.LABEL_SIZE_LIST[-1]
        top_size = ModelConfig.LABEL_SIZE_LIST[0]
        original_la = np.array(original_la)
        fix_part = original_la[:,:-target_size*top_size]
        target_la = original_la[:,-target_size*top_size:]
        
        target_la=np.stack([target_la.reshape([-1, top_size, target_size])[i][actions[i][0]] for i in range(len(actions))],axis=0)

        return np.concatenate([fix_part, target_la],axis=1)

    def _sample_process(self, state_dict, pred_ret):
        '''
        样本处理函数
        '''
        # get is_train
        is_train = False
        req_pb = state_dict['req_pb']
        frame_state = req_pb.frame_state
        for hero in frame_state.hero_states:
            if hero.actor_state.camp == self.hero_camp:
                is_train = True if hero.actor_state.hp > 0 else False

        frame_no = frame_state.frameNo
        feature_vec, reward, sub_action_mask = state_dict['observation'], state_dict['reward'], state_dict['sub_action_mask']
        done = False
        prob, value, action, d_action = pred_ret

        legal_action = self._update_legal_action(state_dict["legal_action"], action)

        keys = ("frame_no", "vec_feature", "legal_action", "action", "reward", "value", "prob", "sub_action",
                "lstm_cell", "lstm_hidden", "done", "is_train")
        
        values = (frame_no, feature_vec, legal_action, action, reward[-1], value, prob, sub_action_mask,
                  self.lstm_cell, self.lstm_hidden, done, is_train)
        sample = dict(zip(keys, values))
        self.last_sample = sample

        if self.save_h5_sample:
            self._sample_process_for_saver(sample)
        return sample

    def _get_h5file_keys(self, h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys

    def _sample_process_for_saver(self, sample_dict):
        keys = ("frame_no", "vec_feature", "legal_action", "action", "reward", "done")
        keys_in_h5 = self._get_h5file_keys(self.dataset)
        
        if len(keys_in_h5) == 0:
            self.dataset.create_dataset("frame_no", data=[[sample_dict["frame_no"]]],
                                        compression="gzip", maxshape=(None, 1), chunks=True)
            self.dataset.create_dataset("observation", data=[sample_dict["vec_feature"]],
                                        compression="gzip", maxshape=(None, len(sample_dict["vec_feature"])), chunks=True)
            self.dataset.create_dataset("legal_action", data=[sample_dict["legal_action"]],
                                        compression="gzip", maxshape=(None, len(sample_dict["legal_action"])), chunks=True)
            self.dataset.create_dataset("action", data=[sample_dict["action"]],
                                        compression="gzip", maxshape=(None, len(sample_dict["action"])), chunks=True)
            self.dataset.create_dataset("reward", data=[[sample_dict["reward"]]],
                                        compression="gzip", maxshape=(None, 1), chunks=True)
            self.dataset.create_dataset("done", data=[[sample_dict["done"]]],
                                        compression="gzip", maxshape=(None, 1), chunks=True)

        else:
            for key, value in sample_dict.items():
                if key in keys:
                    key_dataset = key
                    if key_dataset == "vec_feature":
                        key_dataset = "observation"
                    self.dataset[key_dataset].resize((self.dataset[key_dataset].shape[0] + 1), axis=0)
                    if isinstance(value, list):
                        self.dataset[key_dataset][-1] = value
                    else:
                        self.dataset[key_dataset][-1] = [value]


    def _predict_process(self, feature, legal_action,lstm_cell,lstm_hidden):
        '''
        infer流程，包括数据接口的转换
        '''
        # TODO: add a switch for controlling sample strategy.
        # todo add legal action process, should moved from model _legal_soft_max()
        # put data to input
        
        #batch
        #将5个英雄的数据分离
        batch_size = len(feature)
        input_list = cvt_tensor_to_infer_input(self.model.get_input_tensors())
        input_list[0].set_data(np.stack(feature, axis=0))
        input_list[1].set_data(np.stack(legal_action, axis=0))

        
        input_list[2].set_data(np.stack(lstm_cell, axis=0))
        input_list[3].set_data(np.stack(lstm_hidden, axis=0))

        # todo output should be [self.probs, self.value, self.lstm_cell_output, self.lstm_hidden_output]
        output_list = cvt_tensor_to_infer_output(self.model.get_output_tensors())
        output_list = self._predictor.inference(input_list=input_list, output_list=output_list)
        # cvt output dataxz
        np_output = cvt_infer_list_to_numpy_list(output_list)
        
        logits, value, lstm_cell, lstm_hidden = np_output[:4]
        
        prob, action, d_action = self._sample_masked_action(logits, np.stack(legal_action, axis=0))
        
        arr_size=len(prob)
        batch_size=len(prob[0])
        batch_prob=[]
        batch_action=[]
        batch_d_action=[]
        for i in range(batch_size):
            batch_prob.append([prob[j][i].tolist() for j in range(arr_size)])
            batch_action.append([action[j][i].tolist() for j in range(arr_size)])
            batch_d_action.append([d_action[j][i].tolist() for j in range(arr_size)])
        
        return batch_prob, value, batch_action, batch_d_action, lstm_cell, lstm_hidden  # prob: [[ ]], others: all 1D

    def _sample_masked_action(self, logits, legal_action):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional []
        """
        prob_list = []
        action_list = []
        d_action_list = []
        
        label_split_size = [sum(self.label_size_list[:index + 1]) for index in
                            range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1], axis=1)
        logits_split = np.split(logits, label_split_size[:-1], axis=1)
        
        for index in range(0, len(self.label_size_list)-1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list.append(probs)
            sample_action = self._legal_sample(probs, use_max=False)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

        # deals with the last prediction, target
        index = len(self.label_size_list) - 1
        
        target_legal_action_o = np.reshape(legal_actions[index], # [n,12, 8]
                                           [-1, self.legal_action_size[0],
                                            self.legal_action_size[-1] // self.legal_action_size[0]])
        
        one_hot_actions = np.take(np.eye(self.label_size_list[0]),action_list[0],0)  #[n,12]
        one_hot_actions = np.reshape(one_hot_actions, [-1,self.label_size_list[0], 1]) # [n,12, 1]
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=1)# [n,8]

        legal_actions[index] = target_legal_action # [8]
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list.append(probs)
        sample_action = self._legal_sample(probs, use_max=False)
        action_list.append(sample_action)
        # target_legal_action = tf.gather(target_legal_action, action_idx, axis=1)
        
        one_hot_actions = np.take(np.eye(self.label_size_list[0]),d_action_list[0],0) #[n,12]
        one_hot_actions = np.reshape(one_hot_actions, [-1,self.label_size_list[0], 1]) # [n,12, 1]
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=1)# [n,8]

        # legal_actions[index] = target_legal_action
        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        # prob_list.append(probs)
        d_action = self._legal_sample(probs, use_max=True)
        d_action_list.append(d_action)
        
        return prob_list, action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True,axis=1)
        # Not necessary max clip 1
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        # tmp = tf.exp(tmp - tmp_max)* legal_action + _lsm_const_e
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        # tmp_sum = tf.reduce_sum(tmp, axis=1, keepdims=True)
        probs = tmp / np.sum(tmp, keepdims=True, axis=1)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        """
        Sample with probability, input probs should be 1D array
        """
        if use_max:
            return np.argmax(probs,axis=1)

        res=[]
        for p in probs:
            res.append(np.argmax(np.random.multinomial(1, p, size=1)))
        return res

    def close(self):
        if self.dataset is not None:
            self.save_h5_sample = True
            self.dataset.close()
