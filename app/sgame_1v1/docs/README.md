## 代码说明
## aisrv
aisrv进程， 主要是负责业务逻辑处理  
sgame_run_handler.py，battlesrv和aisrv的交互流程  
sample_processor，样本生产相关逻辑  
protocl，aisrv与battlesrv协议  
lib，特征值处理  

## actor  
actor进程，推理，目前支持TenosorRT和TensorFlow的推理功能

## learner  
learner进程，训练，目前支持TensorFlow的训练功能

## tools  
工具集合，包括启动、停止进程等


## 业务开发流程
## aisrv  
on_update_req，每一帧的处理逻辑  
on_init，初始化操作  
sgame_sample_processor.py，样本格式需要和actor、learner侧的model输入对齐  

上述代码会由KaiwuDRL来调用运行
## actor  
config.py，算法里的配置项修改  
model.py，修改网络结构  
sgame_model.py，训练和预测的流程  
sgame_network.py，训练和预测的流程  

## learner  
config.py，算法里的配置项修改  
model.py，修改网络结构，和actor下的model.py等同，修改一处就行  
sgame_model.py，训练和预测的流程  
sgame_network.py，训练和预测的流程    

## 运行流程  
代码修改完成后，采用tools下的脚本启动进程，查看运行结果


## 注意事项
aisrv目录里的逻辑处理，建议不做修改
