import re
import sys
import os
import numpy as np
from robot.utils import *
from functools import partial

path = os.path.dirname(__file__)
path = "/".join(path.split("\\")[:-1])
sys.path.append(path)


class balance_navi_Robot(object):
    def __init__(self,
                 basePos: list = [0., 0., 0.],
                 baseOri: list = [0., 0., 0., 1.],
                 physicsClientId: int = 0):
        self._physics_client_id = physicsClientId
        # 读入各项参数
        param_path = os.path.join(os.path.dirname(__file__), "config/miniBox_parameters.yaml")
        param_dict = load(open(param_path, "r", encoding="utf-8"), Loader=Loader)
        for key, value in param_dict.items():
            setattr(self, key, value)

        urdf_path = os.path.join(os.path.dirname(__file__), "urdf/balancebot.urdf")
        self.robot = p.loadURDF(
            fileName=urdf_path,
            basePosition=basePos,
            baseOrientation=baseOri,
            useMaximalCoordinates=self.USE_MAX_COOR,
            physicsClientId=physicsClientId
        )

        # 该偏函数用于将输入的速度进行合适的裁剪
        self.clipv = partial(np.clip, a_min=-TARGET_VELOCITY, a_max=TARGET_VELOCITY)

        self.prePos = basePos

    def get_bothId(self):
        return self._physics_client_id, self.robot

    def apply_action(self, action):  # 施加动作
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=[LEFT_WHEEL_JOINT_INDEX, RIGHT_WHEEL_JOINT_INDEX],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[0, 0],
            forces=[0, 0],
        )
        if not (isinstance(action, list) or isinstance(action, np.ndarray)):
            assert f"apply_action() only receive list or ndarray, but receive {type(action)}"
        left_v, right_v = action

        p.setJointMotorControl2(
            bodyUniqueId=self.robot,
            jointIndex=LEFT_WHEEL_JOINT_INDEX,
            controlMode=p.TORQUE_CONTROL,
            force=-left_v
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.robot,
            jointIndex=RIGHT_WHEEL_JOINT_INDEX,
            controlMode=p.TORQUE_CONTROL,
            force=right_v
        )

    def get_balance_obs(self):
        basePos, baseOri = p.getBasePositionAndOrientation(self.robot, physicsClientId=self._physics_client_id)
        baseVel, baseAngVel = p.getBaseVelocity(self.robot, physicsClientId=self._physics_client_id)
        baseEuler = p.getEulerFromQuaternion(baseOri)
        baseEuler = np.array(baseEuler)
        # 归一化euler, 使其在-pi到pi之间

        baseEuler = (baseEuler + np.pi) % (2 * np.pi) - np.pi
        normalized_pitch = baseEuler[0] - (np.pi / 2)
        # normalized_pitch = -baseEuler[0]
        pitch_ang_vel = baseAngVel[0]
        yaw_ang_vel = baseAngVel[2]
        # print("yaw_ang_vel : ", yaw_ang_vel)

        wheel_l_vel = p.getJointState(self.robot, LEFT_WHEEL_JOINT_INDEX, physicsClientId=self._physics_client_id)[1]
        # print("wheel_l_vel : ", p.getJointState(self.robot, LEFT_WHEEL_JOINT_INDEX, physicsClientId=self._physics_client_id))
        wheel_r_vel = p.getJointState(self.robot, RIGHT_WHEEL_JOINT_INDEX, physicsClientId=self._physics_client_id)[1]

        return [normalized_pitch, pitch_ang_vel, yaw_ang_vel, wheel_l_vel, wheel_r_vel]

    def get_navi_obs(self, targetPos):  # 根据目的地的坐标得到机器人目前的状态
        basePos, baseOri = p.getBasePositionAndOrientation(self.robot)
        _, _, results = rayTest(self.robot, self.LASER_LENGTH, self.LASER_NUM)
        lasers_info = [self.LASER_LENGTH if result[0] == -1 else self.__distance(basePos, result[3]) for index, result
                       in enumerate(results)]
        distance = self.__distance(basePos, targetPos)
        angle = self.__angle(
            v1=self.__get_forward_vector(),
            v2=[y - x for x, y in zip(basePos, targetPos)]
        )
        result_lasers = [item / 18.0 for item in lasers_info]
        angle = angle / 3.14
        baseEuler = p.getEulerFromQuaternion(baseOri)
        angle_v = baseEuler[2] / 3.14

        return result_lasers + [distance, angle_v, angle]

    def curPos(self):
        return p.getBasePositionAndOrientation(self.robot)[0]

    def getSpeed(self):
        basePos, baseOri = p.getBasePositionAndOrientation(self.robot)
        speed = self.__distance(basePos, self.prePos)
        self.prePos = basePos
        return speed

    def __get_forward_vector(self):  # 获取机器人朝向的向量
        _, baseOri = p.getBasePositionAndOrientation(self.robot)
        matrix = p.getMatrixFromQuaternion(baseOri)
        return [matrix[0], matrix[3], matrix[6]]

    def __distance(self, v1, v2):
        return sqrt(sum([(x - y) * (x - y) for x, y in zip(v1, v2)]))

    def __angle(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        cosangle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(cosangle)
