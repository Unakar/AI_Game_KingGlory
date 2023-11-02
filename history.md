default:
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