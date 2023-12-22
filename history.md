default_template:
```json
{
  "reward_money": "0.01",
  "reward_exp": "0.01" ,
  "reward_hp_point": "5.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.1",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "1.0",
  "reward_last_hit": "0.5",
  "log_level": "8"
}
```
- conf/configue.ini
  - ppo_clip_range = 0.3（默认0.3）
  - learning_rate = 1e-4（默认1e-4）
  - var_beta = 0.1（默认0.1）
  - production_consume_ratio = 3（默认3）
- conf/learner.ini
  - ppo_epoch = 3（默认3）
- common/configs/config.py
  - GAMMA = 0.995(默认0.995)
  - LAMDA = 0.95(默认0.95)

By wxg
v1:
```json
{
  "reward_money": "0.05",
  "reward_exp": "0.05" ,
  "reward_hp_point": "5.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.1",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "2.0",
  "reward_last_hit": "0.5",
  "log_level": "8"
}
```

By wxg
**v2:**暂时最优
```json
{
  "reward_money": "0.02",
  "reward_exp": "0.02" ,
  "reward_hp_point": "5.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.2",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "3.0",
  "reward_last_hit": "0.5",
  "log_level": "8"
}
```
production_consume_ratio = 1
learning_rate = 1e-4

By wxg
v3:
> 经过对v2败局的分析发现，模型在击败对方英雄后容易待在原地，不积极推兵线，导致经济和总伤害比对方低很多，因此调高了经济的比例，并加大了kill的惩罚与经济系数的增加的对冲
```json
{
  "reward_money": "0.04",
  "reward_exp": "0.04" ,
  "reward_hp_point": "5.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.45",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "3.0",
  "reward_last_hit": "0.5",
  "log_level": "8"
}
```
production_consume_ratio = 1
learning_rate = 1e-4

By wxg
v4:
> 在v3调参中发现了一个winrate达到1的模型，将其取出，降低学习率至原来的1/4，小火收汁.jpg
```json
{
  "reward_money": "0.04",
  "reward_exp": "0.04" ,
  "reward_hp_point": "5.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.45",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "3.0",
  "reward_last_hit": "0.5",
  "log_level": "8"
}
```
production_consume_ratio = 1
learning_rate = 2.5e-5

By wxg
v5:
> 在调高经济后，发现纸面数据已经十分可观，但是由于不积极推塔，即使KDA，经济等有极大优势，仍然输给base4，因此激进地调高了推塔的奖励, 并将lr置到默认的1/2
```json
{
  "reward_money": "0.05",
  "reward_exp": "0.05" ,
  "reward_hp_point": "5.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.5",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "6.0",
  "reward_last_hit": "0.5",
  "log_level": "8"
}
```
production_consume_ratio = 1
learning_rate = 5e-5

By xt
v6
> 尝试调参GAMMA，注重长期收益，学会推塔，以v3作为蓝本，略微调高tower reward
```json
{
  "reward_money": "0.04",
  "reward_exp": "0.04" ,
  "reward_hp_point": "5.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.45",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "4.0",
  "reward_last_hit": "0.5",
  "log_level": "8"
}
```
production_consume_ratio = 1
learning_rate = 8e-5
GAMMA = 0.9975(默认0.995)

By xt
v8
> 尝试调参GAMMA，以v3作为蓝本，略微调高tower reward
```json
{
  "reward_money": "0.04",
  "reward_exp": "0.04" ,
  "reward_hp_point": "4.8",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.45",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "3.5",
  "reward_last_hit": "0.5",
  "log_level": "8"
}
```
production_consume_ratio = 1
learning_rate = 5e-5
clip = 0.25
GAMMA = 0.9975(默认0.995)

By xt
restart v1
> 尝试进行大规模修改调参，主要是提高last_hit的奖励
```json
{
  "reward_money": "0.04",
  "reward_exp": "0.04" ,
  "reward_hp_point": "4.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.45",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "5",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
```
production_consume_ratio = 1
learning_rate = 5e-5
ppo_clip_range = 0.2
GAMMA = 0.996
LAMDA = 0.965

By xt
restart v2
> v1基础上修改batch,lr,hp,tower
```json
{
  "reward_money": "0.04",
  "reward_exp": "0.04" ,
  "reward_hp_point": "3.5",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.45",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "5.5",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
```
batch_size_release = 256(原512)
learning_rate = 3e-5

By wxg
v10 based on v8
> v8的纸面数据已较为客观，但进攻策略过于保守，导致在顺风局被反杀，因此大幅提高tower_hp_point和last_hit使进攻和推塔更激进
{
  "reward_money": "0.05",
  "reward_exp": "0.05" ,
  "reward_hp_point": "5.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.5",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "10.0",
  "reward_last_hit": "3.0",
  "log_level": "8"
}
> 学习率置为默认，样本消耗比置为1，ppo_epoch置为1，GAMMA和LAMDA调高
- conf/configue.ini
  - ppo_clip_range = 0.3（默认0.3）
  - learning_rate = 1e-4（默认1e-4）
  - var_beta = 0.1（默认0.1）
  - production_consume_ratio = 1（默认5）

- conf/learner.ini
  - ppo_epoch = 1（默认3）
- common/configs/config.py
  - GAMMA = 0.996(默认0.995)
  - LAMDA = 0.96(默认0.95)

By wxg
v11 based on v10
> v10推塔和进攻欲望过高，降低了对自身健康的重视，稍微降低塔的奖励和进攻奖励，略微提高法力奖励，防止乱放技能
{
  "reward_money": "0.05",
  "reward_exp": "0.05" ,
  "reward_hp_point": "5.0",
  "reward_ep_rate": "2.0",
  "reward_kill": "-0.5",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "7.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
> 提高GAMMA，注重长期收益，提高clip，快速学习新策略
- conf/configue.ini
  - ppo_clip_range = 0.6（默认0.3）
  - learning_rate = 1e-4（默认1e-4）
  - var_beta = 0.1（默认0.1）
  - production_consume_ratio = 1（默认5）

- conf/learner.ini
  - ppo_epoch = 1（默认3）
- common/configs/config.py
  - GAMMA = 0.9975(默认0.995)
  - LAMDA = 0.96(默认0.95)

By wxg
v12 based on v8
{
  "reward_money": "0.04",
  "reward_exp": "0.04" ,
  "reward_hp_point": "4.8",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.45",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "8.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
- conf/configue.ini
  - ppo_clip_range = 0.2（默认0.3）
  - learning_rate = 5e-5（默认1e-4）
  - var_beta = 0.1（默认0.1）
  - production_consume_ratio = 1（默认5）
- conf/learner.ini
  - ppo_epoch = 1（默认3）
- common/configs/config.py
  - GAMMA = 0.9975(默认0.995)
  - LAMDA = 0.98(默认0.95)

By wxg
t2v1
> 重训练
{
  "reward_money": "0.04",
  "reward_exp": "0.04" ,
  "reward_hp_point": "5.0",
  "reward_ep_rate": "2.0",
  "reward_kill": "-0.4",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "8.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
- conf/configue.ini
  - ppo_clip_range = 0.3（默认0.3）
  - learning_rate = 1e-4（默认1e-4）
  - var_beta = 0.1（默认0.1）
  - production_consume_ratio = 1（默认5）

- conf/learner.ini
  - ppo_epoch = 1（默认3）
- common/configs/config.py
  - GAMMA = 0.9975(默认0.995)
  - LAMDA = 0.98(默认0.95)

By wxg
t2v2
> ep_rate调为默认参数
{
  "reward_money": "0.04",
  "reward_exp": "0.04" ,
  "reward_hp_point": "5.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.4",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "8.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
- conf/configue.ini
  - ppo_clip_range = 0.3（默认0.3）
  - learning_rate = 1e-4（默认1e-4）
  - var_beta = 0.1（默认0.1）
  - production_consume_ratio = 1（默认5）

- conf/learner.ini
  - ppo_epoch = 1（默认3）
- common/configs/config.py
  - GAMMA = 0.9975(默认0.995)
  - LAMDA = 0.98(默认0.95)

By wxg
t3v1
```json
{
  "reward_money": "0.03",
  "reward_exp": "0.03" ,
  "reward_hp_point": "3.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.2",
  "reward_dead": "-0.5",
  "reward_tower_hp_point": "6.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
```
- conf/configue.ini
  - ppo_clip_range = 0.3（默认0.3）
  - learning_rate = 1e-4（默认1e-4）
  - var_beta = 0.1（默认0.1）
  - production_consume_ratio = 3（默认3）
- conf/learner.ini
  - ppo_epoch = 3（默认3）
- common/configs/config.py
  - GAMMA = 0.995(默认0.995)
  - LAMDA = 0.95(默认0.95)

By wxg
t3v2
```json
{
  "reward_money": "0.03",
  "reward_exp": "0.03" ,
  "reward_hp_point": "3.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.2",
  "reward_dead": "-0.5",
  "reward_tower_hp_point": "6.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
```
- conf/configue.ini
  - ppo_clip_range = 0.25（默认0.3）
  - learning_rate = 5e-5（默认1e-4）
  - var_beta = 0.1（默认0.1）
  - production_consume_ratio = 2（默认3）
- conf/learner.ini
  - ppo_epoch = 2（默认3）
- common/configs/config.py
  - GAMMA = 0.996(默认0.995)
  - LAMDA = 0.96(默认0.95)

By wxg
t4v1 based on v3_12h
```json
{
  "reward_money": "0.04",
  "reward_exp": "0.04" ,
  "reward_hp_point": "5.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.45",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "6.0",
  "reward_last_hit": "0.5",
  "log_level": "8"
}
```
- conf/configue.ini
  - ppo_clip_range = 0.3（默认0.3）
  - learning_rate = 1e-4（默认1e-4）
  - var_beta = 0.1（默认0.1）
  - production_consume_ratio = 1（默认3）
- conf/learner.ini
  - ppo_epoch = 3（默认3）
- common/configs/config.py
  - GAMMA = 0.996(默认0.995)
  - LAMDA = 0.96(默认0.95)

By wxg
t4v2 based on t4v1_9h
```json
{
  "reward_money": "0.04",
  "reward_exp": "0.04" ,
  "reward_hp_point": "5.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.45",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "6.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
```
- conf/configue.ini
  - ppo_clip_range = 0.3（默认0.3）
  - learning_rate = 5e-5（默认1e-4）
  - var_beta = 0.1（默认0.1）
  - production_consume_ratio = 1（默认3）
- conf/learner.ini
  - ppo_epoch = 1（默认3）
- common/configs/config.py
  - GAMMA = 0.996(默认0.995)
  - LAMDA = 0.96(默认0.95)

By wxg
t4v3 based on t4v2_12h
```json
{
  "reward_money": "0.04",
  "reward_exp": "0.04" ,
  "reward_hp_point": "5.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.45",
  "reward_dead": "-1.0",
  "reward_tower_hp_point": "6.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
```
- conf/configue.ini
  - ppo_clip_range = 0.2（默认0.3）
  - learning_rate = 2.5e-5（默认1e-4）
  - var_beta = 0.08（默认0.1）
  - production_consume_ratio = 1（默认3）
- conf/learner.ini
  - ppo_epoch = 1（默认3）
- common/configs/config.py
  - GAMMA = 0.9965(默认0.995)
  - LAMDA = 0.965(默认0.95)

By wxg
t4v4 based on t4v3_12h
```json
{
  "reward_money": "0.03",
  "reward_exp": "0.03" ,
  "reward_hp_point": "3.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.2",
  "reward_dead": "-0.5",
  "reward_tower_hp_point": "6.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
```
- conf/configue.ini
  - ppo_clip_range = 0.2（默认0.3）
  - learning_rate = 2.5e-5（默认1e-4）
  - var_beta = 0.05（默认0.1）
  - production_consume_ratio = 1（默认3）
- conf/learner.ini
  - ppo_epoch = 1（默认3）
- common/configs/config.py
  - GAMMA = 0.997(默认0.995)
  - LAMDA = 0.97(默认0.95)

By wxg
t4v5 based on t4v4_12h
```json
{
  "reward_money": "0.03",
  "reward_exp": "0.03" ,
  "reward_hp_point": "3.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.2",
  "reward_dead": "-0.5",
  "reward_tower_hp_point": "6.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
```
- conf/configue.ini
  - ppo_clip_range = 0.15（默认0.3）
  - learning_rate = 1e-5（默认1e-4）
  - var_beta = 0.03（默认0.1）
  - production_consume_ratio = 1（默认3）
- conf/learner.ini
  - ppo_epoch = 1（默认3）
- common/configs/config.py
  - GAMMA = 0.9975(默认0.995)
  - LAMDA = 0.975(默认0.95)

By wxg
t4v6 based on t4v5_12h
```json
{
  "reward_money": "0.03",
  "reward_exp": "0.03" ,
  "reward_hp_point": "3.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.2",
  "reward_dead": "-0.5",
  "reward_tower_hp_point": "6.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
```
- conf/configue.ini
  - ppo_clip_range = 0.1（默认0.3）
  - learning_rate = 5e-6（默认1e-4）
  - var_beta = 0.02（默认0.1）
  - production_consume_ratio = 1（默认3）
- conf/learner.ini
  - ppo_epoch = 1（默认3）
- common/configs/config.py
  - GAMMA = 0.998(默认0.995)
  - LAMDA = 0.98(默认0.95)

By wxg
t5v1 based on t4v2_12h
```json
{
  "reward_money": "0.03",
  "reward_exp": "0.03" ,
  "reward_hp_point": "3.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.2",
  "reward_dead": "-0.5",
  "reward_tower_hp_point": "6.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
```
- conf/configue.ini
  - ppo_clip_range = 0.3（默认0.3）
  - learning_rate = 5e-5（默认1e-4）
  - var_beta = 0.1（默认0.1）
  - production_consume_ratio = 1（默认3）
- conf/learner.ini
  - ppo_epoch = 1（默认3）
- common/configs/config.py
  - GAMMA = 0.996(默认0.995)
  - LAMDA = 0.96(默认0.95)

By wxg
t5v2 based on t5v1_12h
```json
{
  "reward_money": "0.03",
  "reward_exp": "0.03" ,
  "reward_hp_point": "3.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.2",
  "reward_dead": "-0.5",
  "reward_tower_hp_point": "6.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
```
- conf/configue.ini
  - ppo_clip_range = 0.2（默认0.3）
  - learning_rate = 2.5e-5（默认1e-4）
  - var_beta = 0.05（默认0.1）
  - production_consume_ratio = 1（默认3）
- conf/learner.ini
  - ppo_epoch = 1（默认3）
- common/configs/config.py
  - GAMMA = 0.9965(默认0.995)
  - LAMDA = 0.965(默认0.95)

By wxg
t5v3 based on t5v2_12h
```json
{
  "reward_money": "0.03",
  "reward_exp": "0.03" ,
  "reward_hp_point": "3.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.2",
  "reward_dead": "-0.5",
  "reward_tower_hp_point": "6.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
```
- conf/configue.ini
  - ppo_clip_range = 0.15（默认0.3）
  - learning_rate = 1e-5（默认1e-4）
  - var_beta = 0.03（默认0.1）
  - production_consume_ratio = 1（默认3）
- conf/learner.ini
  - ppo_epoch = 1（默认3）
- common/configs/config.py
  - GAMMA = 0.997(默认0.995)
  - LAMDA = 0.97(默认0.95)

By wxg
t6v1 based on t5v2_12h
```json
{
  "reward_money": "0.03",
  "reward_exp": "0.03" ,
  "reward_hp_point": "3.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.2",
  "reward_dead": "-0.5",
  "reward_tower_hp_point": "6.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
```
- conf/configue.ini
  - ppo_clip_range = 0.2（默认0.3）
  - learning_rate = 2.5e-5（默认1e-4）
  - var_beta = 0.05（默认0.1）
  - production_consume_ratio = 1（默认3）
- conf/learner.ini
  - ppo_epoch = 1（默认3）
- common/configs/config.py
  - GAMMA = 0.9965(默认0.995)
  - LAMDA = 0.965(默认0.95)

By wxg
t6v2 based on t6v1_12h
```json
{
  "reward_money": "0.03",
  "reward_exp": "0.03" ,
  "reward_hp_point": "3.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.2",
  "reward_dead": "-0.5",
  "reward_tower_hp_point": "6.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
```
- conf/configue.ini
  - ppo_clip_range = 0.2（默认0.3）
  - learning_rate = 2.5e-5（默认1e-4）
  - var_beta = 0.05（默认0.1）
  - production_consume_ratio = 2（默认3）
  - batch_size_release = 128（默认512）
- conf/learner.ini
  - ppo_epoch = 3（默认3）
- common/configs/config.py
  - GAMMA = 0.996(默认0.995)
  - LAMDA = 0.965(默认0.95)

By xt
t6v2 based on t6v1
```json
{
  "reward_money": "0.03",
  "reward_exp": "0.03" ,
  "reward_hp_point": "3.0",
  "reward_ep_rate": "0.75",
  "reward_kill": "-0.2",
  "reward_dead": "-0.6",
  "reward_tower_hp_point": "6.0",
  "reward_last_hit": "2.0",
  "log_level": "8"
}
```
- conf/configue.ini
  - ppo_clip_range = 0.2（默认0.3）
  - learning_rate = 2.5e-5（默认1e-4）
  - var_beta = 0.05（默认0.1）
  - production_consume_ratio = 1（默认3）
- conf/learner.ini
  - ppo_epoch = 1（默认3）
- common/configs/config.py
  - GAMMA = 0.995
  - LAMDA = 0.965(默认0.95)