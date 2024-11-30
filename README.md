# Balance_navi

## Balance_naviV2.0
### This is version2.0, date 2024-11-19
##### 1、更改LSTM
将LSTM相关的代码修改为导航模块的LSTM
受影响文件：\RL_algorithm\Actor.py&Critic.py&DDPG.py&ReplayBuffer.py

##### 2、更改测试时间
将每轮测试的步长修改
受影响文件：\robot\config\task.yaml

##### 3、修改scene
把scene换成了导航的scene

##### 4、删除多余文件
删除导航部分的多余文件
如 \robot\minibox.py

##### 5、重写main文件
首先将导航和平衡的功能函数整合进\robot\balancebot.py，如获取导航和平衡的obs。
在\robot\utils.py中把雷达的方向旋转180°。
最后重写控制逻辑，将导航模型输出navi_action的左轮速进行取反，然后将整体action限制范围，加到balance_action上，进行step
