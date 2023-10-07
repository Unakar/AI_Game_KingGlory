#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# 模块介绍：本模块封装了模型的创建和训练操作，封装了模型与环境交互的操作

'''
@Project :kaiwu-fwk 
@File    :game_controller.py
@Author  :kaiwu
@Date    :2022/6/15 20:57 

'''

import os
import random
import numpy as np
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.util import nest
from tensorflow.python.ops import data_flow_ops
from framework.common.utils.tf_utils import *
from framework.server.python.learner.gradient_fusion import *
from app.sgame_1v1.common.models.model import Model
from app.sgame_1v1.actor_learner.agent import Agent
from framework.common.utils.common_func import TimeIt, get_local_rank

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class Graphs(object):
    '''
        模型的封装类，build_model和_add_forward_pass_and_gradients是核心逻辑
        获取tf.ops
    '''
    def __init__(self, network):
        self.batch_size = CONFIG.train_batch_size
        self.cpu_device = '/cpu:0'

        if tf.test.is_built_with_cuda():
            self.device = '/gpu:0'
        else:
            self.device = self.cpu_device

        if CONFIG.use_fp16:
            self.key_types = [tf.float16]
        else:
            self.key_types = [tf.float32]
        self.network = network
        self.gradient_fusion = GradientFusion()
        self.loss_has_inf_nan = None
        self.grad_has_inf_nan = None

    def get_data_list_shape(self, data_list):
        list_shapes = []
        for i in range(len(data_list)):
            list_shapes.append(data_list[i].get_shape())
        return list_shapes

    def build_model(self, input_datas):
        '''
        创建模型的主要函数
        '''
        enqueue_ops = list()
        fetches = dict()
        training_ops = list()

        with tf.device(self.cpu_device):
            global_step = tf.train.get_or_create_global_step()
            self.global_step = global_step
            datas = input_datas

        with tf.variable_scope('', reuse=tf.AUTO_REUSE), tf.name_scope('tower_0') as name_scope:
            with tf.xla.experimental.jit_scope(CONFIG.use_xla):
                loss, info_list, gradvars, max_noisescale, gpu_copy_stage_op, gpu_compute_stage_op = \
                    self._add_forward_pass_and_gradients(datas)

            enqueue_ops.append(gpu_copy_stage_op)
            enqueue_ops.append(gpu_compute_stage_op)

            fetches['enqueue_ops'] = enqueue_ops
            fetches['info_list'] = info_list
            fetches['noise_scale'] = max_noisescale

            with tf.device(self.device):
                if not CONFIG.enable_mixed_precision:
                    self.opt = self.network.get_optimizer()

                training_ops.append([self.opt.apply_gradients(gradvars)])
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
                train_op = tf.group(*(training_ops + update_ops))

            fetches['train_op'] = train_op
            fetches['total_loss'] = loss
            if self.grad_has_inf_nan is not None:
                fetches['grad_has_inf_nan'] = self.grad_has_inf_nan
            if self.loss_has_inf_nan is not None:
                fetches['train_has_inf_nan'] = [self.loss_has_inf_nan, self.grad_has_inf_nan]

        return (enqueue_ops, fetches)

    def _check_grads(self, grads):
        has_inf_nan_list = []
        for (grad, _) in grads:
            has_inf_nan_list.append(tf.cast(tf.reduce_sum(tf.cast(tf.is_inf(grad), dtype=tf.int32)), dtype=tf.bool))
            has_inf_nan_list.append(tf.cast(tf.reduce_sum(tf.cast(tf.is_nan(grad), dtype=tf.int32)), dtype=tf.bool))
        self.grad_has_inf_nan = tf.reduce_all(has_inf_nan_list)
        assert self.grad_has_inf_nan is not None
  
    def _add_forward_pass_and_gradients(self, datas):
        '''
            创建存储间拷贝的ops，loss、gradient等ops
        '''
        with tf.device(self.cpu_device):
            gpu_copy_stage = data_flow_ops.StagingArea(self.key_types, \
                                                       shapes=self.get_data_list_shape(datas))
            gpu_copy_stage_op = gpu_copy_stage.put(datas)
            datas = gpu_copy_stage.get()

        with tf.device(self.device):
            gpu_compute_stage = data_flow_ops.StagingArea(self.key_types, \
                                                          shapes=self.get_data_list_shape(datas))
            gpu_compute_stage_op = gpu_compute_stage.put(datas)
            datas = gpu_compute_stage.get()

        with tf.device(self.device):
            loss, info_list = self.network.build_graph(datas[0], self.global_step)
            params = tf.trainable_variables()
            aggmeth = tf.AggregationMethod.DEFAULT

            if CONFIG.enable_mixed_precision:
                self.opt = self.network.get_optimizer()
                self.opt = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(self.opt)
                gradvars = self.opt.compute_gradients(loss)
            else:
                grads = tf.gradients(loss, params, aggregation_method=aggmeth)
                gradvars = list(zip(grads, params))

            if CONFIG.check_values:
                self.loss_has_inf_nan = tf.logical_or(tf.is_inf(loss), tf.is_nan(loss))
                assert self.loss_has_inf_nan is not None
                self._check_grads(gradvars)

            gradvars, max_noisescale = self.gradient_fusion.run(gradvars)

            return loss, info_list, gradvars, max_noisescale, gpu_copy_stage_op, gpu_compute_stage_op


class GameController(object):
    '''
        GameController是sgame_1v1的具体算法类, actor和learner都会调用
        由于两者行为部分行为不同, 采用model参数来进行区分
    '''

    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 8]

    '''
        Model类的初始化函数
        Args:
            network: sgame_1v1自定义Network类的实例
            name: 模型名称
            mode: 使用模式, actor或者Learner
    '''
    def __init__(self, network=None, name="sgame", mode="actor"):
        assert mode in ["actor", "learner"]
        self.mode = mode
        self.name = name
        self.cpu_device = '/cpu:0'
        if tf.test.is_built_with_cuda():
            self.device = '/gpu:0'
        else:
            self.device = self.cpu_device

        self.model_pool_addr = CONFIG.modelpool_remote_addrs

        self.category_names = ["which_button", "move_x", "move_z", "kill_x", "kill_z", "target"]

        self.action_size = self.LABEL_SIZE_LIST
        self.legal_action_split_size = []
        tmp = 0
        for i, s in enumerate(self.action_size):
            tmp += s
            self.legal_action_split_size.append(tmp)
        self.legal_action_split_size = self.legal_action_split_size
        
        if mode == "learner":
            hvd.init()

            self.node_info = NodeInfo()
            self.is_chief_rank = self.node_info.rank == 0

            self.graph = Graphs(Model(mode=mode))
            self.name = name
            self.local_step = 0
            self.step_train_times = list()
            self.total_noise_scale = 0.0
            self.noise_scale_times = 0
            self.skip_update_times = 0

            self.sess = None
            self.first_init_enqueue_ops = False
        else:
            assert network is not None

    def set_dataset(self, dataset):
        self.dataset = dataset

    def build_model(self, *args, **kwargs):
        if self.mode == "actor":
            self.build_actor_model(*args, **kwargs)
        else:
            self.build_learner_model()

    def build_actor_model(self, *args, **kwargs):
        '''
            actor侧调用
        '''
        self.reset()
        self.logger.info('sgame_actor model build success')
    
    def reset(self):
        with tf.device(self.device):
            self.on_init()

    def on_init(self):
        '''
            游戏还没有启动，进行一些初始化操作
        '''

        self.logger.info("sgame_actor init start")

        self.first = True
        self.is_gameover = False
        self.agent_num = 1

        self.create_agent()

        # TODO:完善红蓝方逻辑
        # self.agents.reverse() 

        self.actions = []
        self.rewards = [[], []]
        self.step = 0
        self.game_id = "1111111"

        for i, agent in enumerate(self.agents):
            agent.reset("network")

        self.logger.info("sgame_actor init success")

    def create_agent(self):
        '''
            需要确保启动了modelpoll
        '''
        agents = []
        main_agent = random.randint(0, 1)
        for i in range(self.agent_num):
            agents.append(Agent(
                Model, self.model_pool_addr,
                keep_latest=(True),  # TODO：设置为True
                local_mode=False,
                # dataset=dataset_f  # comment this out when collecting data
            ))
        
        self.agents = agents

        self.logger.info(f'sgame_actor create_agent {self.agents}')

    def build_learner_model(self):
        with tf.device(self.cpu_device):
            input_datas = self.dataset.dataset_from_generator()

        (self.enqueue_ops, self.fetches) = self.graph.build_model(input_datas)
        info_list = self.fetches['info_list']
        fetches_list = nest.flatten(list(self.fetches.values()))
        main_fetch_group = tf.group(*fetches_list)

        with tf.device(self.cpu_device):
            self.global_step = tf.train.get_global_step()
            self.fetches["global_step"] = self.global_step
            with tf.control_dependencies([main_fetch_group]):
                self.fetches['inc_global_step'] = self.global_step.assign_add(1)

                # 增加summary的展示
                with tf.compat.v1.variable_scope("summary"):
                    tf.compat.v1.summary.scalar("loss_all", info_list[0]),
                    tf.compat.v1.summary.scalar("value_loss", info_list[1]),
                    tf.compat.v1.summary.scalar("policy_loss", info_list[2]),
                    tf.compat.v1.summary.scalar("entropy_loss", info_list[3]),
                    tf.compat.v1.summary.scalar("ratio_mean", info_list[4]),
                    tf.compat.v1.summary.scalar("policy_log_p_mean", info_list[5]),
                    tf.compat.v1.summary.scalar("old_policy_log_p_mean", info_list[6]),

                self.fetches['summary_op'] = tf.compat.v1.summary.merge_all()

        self.logger.info('sgame_learner model build success')
    
    def create_config_proto(self, gpu_index):
        '''
            配置函数
        '''
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 2
        config.inter_op_parallelism_threads = 0
        config.gpu_options.force_gpu_compatible = True
        config.gpu_options.visible_device_list = str(gpu_index)
        config.graph_options.rewrite_options.arithmetic_optimization = rewriter_config_pb2.RewriterConfig.OFF
        return config

    def init_model(self):
        # 如果非learner则跳过
        if self.mode != "learner":
            return

        bcast_global_variables_op = self.node_info.get_bcast_op()

        # TensorFlow官方推荐采用 tf.compat.v1.train.MonitoredTrainingSession

        # 保存的最大checkpoint数量
        scaffold = tf.compat.v1.train.Scaffold(saver=tf.compat.v1.train.Saver(max_to_keep=CONFIG.max_to_keep_ckpt_file_num))

        self.sess = tf.compat.v1.train.MonitoredTrainingSession(
            master='',
            checkpoint_dir=f'{CONFIG.restore_dir}/{self.name}/' if self.is_chief_rank else None,
            # summary_dir=f'{CONFIG.summary_dir}/{self.name}/' if self.is_chief_rank else None,
            # hooks=[summary_hook],
            save_checkpoint_secs=CONFIG.save_checkpoint_secs,
            save_checkpoint_steps=None,
            save_summaries_secs=None,
            save_summaries_steps=None,
            chief_only_hooks=None,
            is_chief=True,
            scaffold=scaffold,
            config=self.create_config_proto(self.node_info.local_rank),
        )

        if self.is_chief_rank:
            self.train_writer = tf.summary.FileWriter(f'{CONFIG.summary_dir}/{self.name}/', self.sess.graph)

        if bcast_global_variables_op:
            self.sess.run(bcast_global_variables_op)

        self.sess.run(self.dataset.extra_initializer_ops())

        self.init_global_step = self.sess.run(self.global_step)

        # 打印网络值和权重
        print_variables(self.sess, 'learner')
        self.logger.info(f'sgame_learner print_variables success')

        # 如果需要打开TensorFlow Profile
        if CONFIG.print_profile:
            self.profiler = model_analyzer.Profiler(self.sess.graph)

        self.logger.info(f'sgame_learner init_model success, init_global_step {self.init_global_step}')

    def timeline_after_run(self, step, steps, run_metadata):
        '''
            timeline 后的操作
        '''
        if step in steps:
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(f'{CONFIG.log_dir}/timeline_{step}.json', 'w') as f_timeline:
                f_timeline.write(ctf)

            self.logger.info(f'sgame_learner {CONFIG.log_dir}/timeline_{step}.json success')

            # 本次执行完成后, 设置print_timeline为False
            if step == steps[-1]:
                CONFIG.print_timeline = False

    def proflie_after_run(self, step, run_options, run_metadata, trim_name_regexes=None):
        '''
            profile 后的操作
        '''
        if step in range(CONFIG.print_profile_start_step, CONFIG.print_profile_end_step):
            self.profiler.add_step(step, run_metadata)

        if step >= CONFIG.print_profile_end_step:
            builder = option_builder.ProfileOptionBuilder()
            builder.with_timeline_output(timeline_file=f'{CONFIG.log_dir}/timeline_{get_local_rank()}.json')
            self.profiler.profile_graph(builder.build())
            self.logger.info(f'sgame_learner write profile to {CONFIG.log_dir}/timeline_{get_local_rank()}.json')

            builder = option_builder.ProfileOptionBuilder(
                option_builder.ProfileOptionBuilder.time_and_memory())
            if trim_name_regexes:
                builder.with_node_names(trim_name_regexes=trim_name_regexes)

            builder.order_by('micros')
            builder.with_file_output(outfile=f'{CONFIG.log_dir}/time_and_memory_{get_local_rank()}.txt')
            self.profiler.profile_name_scope(builder.build())
            self.logger.info(f'sgame_learner write profile to {CONFIG.log_dir}/time_and_memory_{get_local_rank()}.txt')

            # 本次执行完成后, 设置print_profile为False
            CONFIG.print_profile = False

    def predict(self, state_dict):
        '''
            Actor的前向传播, 进行预测
            Args:
                state_dict: 预测需要的状态信息,由sgame_1v1的state类定义

            Returns: format_actions, network_sample_info

        '''
        assert self.mode == "actor"
        format_action, network_sample_info, lstm_info = self.on_predict(**state_dict)
        return format_action, network_sample_info, lstm_info
    
    def on_predict(self, observation, legal_action, sub_action_mask, lstm_hidden, lstm_cell, **kwargs):
        '''
            实际的调用预测函数
        '''
        ###输入并没有考虑多agent？
        self.cur_state= []

        #TO DO:多个agent加入多个
        self.cur_state.append({
            "features": observation,
            "legal_action": legal_action,
            "sub_action_mask": sub_action_mask,
        })
        bs = len(observation)

        with TimeIt() as ti:
            network_sample_info = []
            lstm_info = []
            actions = []
            for i, agent in enumerate(self.agents):
                with tf.device(self.device):
                    pred_ret = agent._predict_process(observation, legal_action, lstm_cell, lstm_hidden)
                prob, value, action, d_action, lstm_cell_return, lstm_hidden_return = pred_ret
                if CONFIG.run_mode == "eval":
                    action = d_action
                actions.append(action)
                a, b = self.get_must_sample_info(agent,pred_ret,legal_action)
                network_sample_info.append(a)
                lstm_info.append(b)
            format_actions = self.step_actions(actions)
        
        fs = []
        ns = []
        lstms = []
        for i in range(bs):
            fs.append(tuple([format_actions[j][i] for j, agent in enumerate(self.agents)]))
            ns.append(tuple([network_sample_info[j][i] for j, agent in enumerate(self.agents)]))
            lstms.append(tuple([lstm_info[j][i] for j, agent in enumerate(self.agents)]))
        
        format_actions = fs
        network_sample_info = ns
        lstm_info = lstms
        
        return format_actions, network_sample_info, lstm_info
    
    def step_actions(self, actions):
        self._check_action(actions)
        format_actions = self._format_actions(actions)
        return format_actions

    def _split_legal_action(self, la, button):
        batch_size=len(la)
        tmp = np.split(np.stack(la,axis=0), self.legal_action_split_size[:-1], axis=1)
        
        tmp[-1] = np.stack( [tmp[-1].reshape(batch_size, -1, self.LABEL_SIZE_LIST[-1])[i][b] for i,b in enumerate(button)], axis=0)
        
        return tmp

    def _check_action(self, actions):
        # check whether the actions are legal

        for i, act in enumerate(actions):
            legal = self.cur_state[i]['legal_action']
            legal = self._split_legal_action(legal, np.array(act)[:,0])
            sub = [self.cur_state[i]['sub_action_mask'][j][b] for j,b in enumerate(np.array(act)[:,0])]

            batch_size=len(sub)
            for j in range(6):
                for k in range(batch_size):
                    a = int(act[k][j])
                    if (a < 0 or a >= len(legal[j][k])) or legal[j][k][a] == 0:
                        self.logger.warn(
                            'sgame_learner Agent[{}] is passed with an illegal action {} No.{}:[{}], legal: {}, all: {}, sub: {}'.format(
                                i, self.category_names[j], j, a, legal[j][k], act[k], sub[k]
                            ))

    def _format_actions(self, actions):
        # check whether the actions are within defined range, and format into gamecore actions
        rp_actions = []
        for i, action in enumerate(actions):
            # formulation check
            bs=len(action)
            action=np.stack(action)
            
            if isinstance(action, (tuple, list)):
                if not len(action) == 6:
                    assert False, "action[{}] length incorrect: {}, but expect 6."
                #action = np.array(action)
            elif isinstance(action, np.ndarray):
                if not (len(action.shape) == 2 and action.shape[1] == 6):
                    assert False, "action[{}] shape incorrect: {}, but expect [6].".format(i, action.shape)
            else:
                assert False, "invalid action[{}] type of {}".format(i, type(action))
            
            rp_action=[]
            for k in range(bs):
                acts=action[k]
                old_acts = acts
                acts = []
                for j, act in enumerate(old_acts):
                    assert 0 <= act < self.action_size[j], \
                        "Action[{}] {},batch{}: {} not in [0,{})".format(i, j, k, act, self.action_size[j])
                    acts.append((act,))
                acts = tuple(acts)
                
                # todo change hard code?
                rp_act = ((0,) * 12, (0,) * 16, (0,) * 16, (0,) * 16, (0,) * 16, (0,) * 8,) + \
                        acts + \
                        ((0,),)
                rp_action.append(rp_act)
            rp_actions.append(rp_action)
        return rp_actions

    def get_must_sample_info(self, agent, pred_ret, state_legal_action):

        #获得产生样本所需要的信息
        
        prob, value, action, d_action, lstm_cell, lstm_hidden = pred_ret
        legal_action = agent._update_legal_action(state_legal_action,action)
        
        bs = len(prob)
        res = []
        res_lstm = []
        for i in range(bs):
            res.append((legal_action[i], action[i], value[i], prob[i], lstm_cell[i], lstm_hidden[i]))
            res_lstm.append((lstm_cell[i],lstm_hidden[i]))
        
        return res,res_lstm

    def load_last_new_model(self, models_path):
        '''
            加载最新的模型文件
        '''
        self.on_load_last_new_model(models_path)
    
    def on_load_last_new_model(self, models_path):
        for agent in self.agents:
            agent.load_last_new_model(models_path)

    def train(self):
        '''
            Learner的训练函数
            Returns: None
        '''

        assert self.mode == "learner"
        self.logger.debug('sgame_learner Start training')

        if not self.first_init_enqueue_ops:
            self.sess.run(self.enqueue_ops)
            self.first_init_enqueue_ops = True
            self.logger.info(f'sgame_learner first run init enqueue ops success')

        batch_duration = 0

        run_options = None
        run_metadata = None

        with TimeIt() as ti:
            # 增加样本利用率，同时ppo在当前的onpolicy采样模式下更适合多次更新
            ppo_epoch = int(CONFIG.ppo_epoch)
            for _ in range(ppo_epoch):
                # 在主learner上配置print_timeline/print_profile, 在开发测试阶段设置为True, 线上运用设置为False
                if self.is_chief_rank:
                    print_timeline_or_profile = False
                    if CONFIG.print_timeline:
                        # 这里是一个字符串数组, 需要转换为整形数组
                        local_step_count_when_print_timeline = CONFIG.local_step_count_when_print_timeline.split(',')
                        local_step_count_when_print_timeline = list(map(int, local_step_count_when_print_timeline))
                        if self.local_step in local_step_count_when_print_timeline:
                            print_timeline_or_profile = True
                    if CONFIG.print_profile and self.local_step in range(CONFIG.print_profile_start_step, CONFIG.print_profile_end_step):
                        print_timeline_or_profile = True

                    # 这里设置为run options full trace
                    if print_timeline_or_profile:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()

                    results = self.sess.run(self.fetches, options=run_options, run_metadata=run_metadata)

                    self.train_writer.add_summary(results['summary_op'], self.local_step)
                    self.train_writer.flush()
                    # self.logger.debug(f'summary flush success')

                    if CONFIG.print_timeline and print_timeline_or_profile:
                        self.timeline_after_run(self.local_step, local_step_count_when_print_timeline, run_metadata)

                    if CONFIG.print_profile and print_timeline_or_profile:
                        self.proflie_after_run(self.local_step, run_options, run_metadata, trim_name_regexes=['gradients'])

                else:
                    results = self.sess.run(self.fetches, options=run_options, run_metadata=run_metadata)

        batch_duration = ti.interval
        # self.logger.debug('sgame_learner train cost {} ms'.format(batch_duration * 1000))
        self.step_train_times.append(batch_duration)

        self.local_step += 1
        if self.is_chief_rank and \
                (self.local_step == 0 or self.local_step % CONFIG.display_every == 0):
            results['ip'] = CONFIG.ip_address
            results['batch_size'] = CONFIG.train_batch_size
            results['step'] = self.local_step
            results['gpu_nums'] = self.node_info.size
            results['sample_recv_speed'] = self.dataset.get_recv_speed()
            results['sample_consume_speed'] = self.get_sample_consume_speed( \
                CONFIG.train_batch_size, self.step_train_times)

            self.logger.info(f'sgame_learner run result {results}')
        
        self.logger.debug('sgame_learner end training')

        # 保存model文件是放在单独的进程里处理的

        return results['info_list']
   
    def should_stop(self):
        '''
            游戏环境确定确定是否可以停止, 需要按照需求设置
        '''
        return False

    def set_logger(self, logger):
        '''
            框架提供了日志接口, 框架使用者直接使用即可
        '''
        self.logger = logger

    def stop(self):
        assert self.mode == "learner"
        self.sess.close()

        self.logger.info('sgame_learner self.sess stop success')

    def get_sample_consume_speed(self, batch_size, step_train_times, scale=1):
        '''
            采用新版本的监控获取方式
        '''
        return 0

        if not step_train_times:
            return ''
        
        times = np.array(step_train_times)
        speed_mean = scale * batch_size / np.mean(times)

        # 清理self.step_train_times, 以防队列太大影响计算耗时
        self.step_train_times.clear()
        
        return speed_mean

    def get_global_step(self):
        return self.sess.run(self.global_step)

    @property
    def tf_sess(self):
        return self.sess
