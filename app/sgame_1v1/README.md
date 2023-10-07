## 欢迎来到王者荣耀

### 实验介绍：

**场景：** 王者荣耀1v1

**地图：** 墨家机关道

**阵容：** 鲁班

**支持框架：** TensorFlow

**支持算法：** PPO

### 代码目录介绍

**actor_learner：** 对应于强化学习里的Agent，其中Actor负责推理，Learner负责训练。

**common：** 包含了Actor/Learner都会调用的model文件，以及model的配置文件configs。

**env：** 对应于强化学习里的Environment，负责连接Actor/Learner与Gamecore，主要负责业务逻辑。

**tools：** 工具集合，包括启动、停止进程以及查看运行结果的脚本。

**docs：** 简单的代码说明

💡[点此查看王者荣耀1v1实验开发指南](https://doc.aiarena.tencent.com/edu/hok1v1/latest/hok1v1_guidebook/intro/)