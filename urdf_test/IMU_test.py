import pybullet as p
import pybullet_data
import time
import numpy as np

import sys
import os

path = os.path.dirname(__file__)
path = "/".join(path.split("\\")[:-1])
sys.path.append(path)

# 连接到PyBullet仿真环境
p.connect(p.GUI)

# 设置搜索路径，确保PyBullet能找到URDF文件和相关模型文件
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 加载平面模型
p.loadURDF("plane.urdf")

# 加载你的URDF模型

urdf_path = os.path.join(os.path.dirname(__file__), "urdf/balancebot.urdf")
robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])

# 获取机器人的基座链接索引
base_link_index = 0  # 通常基座链接的索引为0，但请根据你的URDF文件确认

# 运行仿真
while True:
    p.stepSimulation()

    # 获取基座链接的状态
    state = p.getLinkState(robot_id, base_link_index, computeLinkVelocity=1)

    # 获取线速度和角速度
    linear_velocity = state[6]  # [vx, vy, vz]
    angular_velocity = state[7]  # [wx, wy, wz]

    # 打印速度数据
    print("Linear Velocity:", np.array(linear_velocity))
    print("Angular Velocity:", np.array(angular_velocity))

    # 计算加速度（简单的数值微分）
    previous_linear_velocity = np.array(linear_velocity)
    previous_angular_velocity = np.array(angular_velocity)

    # time.sleep(1./240.)

    current_state = p.getLinkState(robot_id, base_link_index, computeLinkVelocity=1)
    current_linear_velocity = np.array(current_state[6])
    current_angular_velocity = np.array(current_state[7])

    linear_acceleration = (current_linear_velocity - previous_linear_velocity) / (1. / 240.)
    angular_acceleration = (current_angular_velocity - previous_angular_velocity) / (1. / 240.)

    # 打印加速度数据
    print("Linear Acceleration:", linear_acceleration)
    print("Angular Acceleration:", angular_acceleration)

    # 更新速度
    previous_linear_velocity = current_linear_velocity
    previous_angular_velocity = current_angular_velocity

    # time.sleep(1./240.)

# 断开连接
p.disconnect()
