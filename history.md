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