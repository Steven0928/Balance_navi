# encoding: utf-8
# 展示和测试文件夹中的urdf文件
# 路径添加
import sys
import os
import time

path = os.path.dirname(__file__)
path = "/".join(path.split("\\")[:-1])
sys.path.append(path)

import pybullet as p
import pybullet_data
from time import sleep
from math import tan, sin, cos, sqrt, acos
import numpy as np
from pprint import pprint
from yaml import load, Loader

# constant var
UP = p.B3G_UP_ARROW
DOWN = p.B3G_DOWN_ARROW
LEFT = p.B3G_LEFT_ARROW
RIGHT = p.B3G_RIGHT_ARROW
SPACE_CLS = p.B3G_SPACE

R2D2_POS = [1., 1., 1.]
ROBOT_POS = [0., 0., 0.2]
DOOR_POS = [-2, 0, 0]

LEFT_WHEEL_JOINT_INDEX = 1
RIGHT_WHEEL_JOINT_INDEX = 0
MAX_FORCE = 10.
TARGET_VELOCITY = 210.
MULTIPLY = 2.0

DEBUG_TEXT_COLOR = [0., 0., 0.]  # debug文本的颜色
DEBUG_TEXT_SIZE = 1.2  # debug文本的大小
MISS_COLOR = [0., 1., 0.]  # 没有命中的激光的颜色
HIT_COLOR = [1., 0., 0.]  # 命中的激光的颜色
RAY_DEBUG_LINE_WIDTH = 2.  # 激光的debug线的宽度

# 机器人参数
BASE_THICKNESS = 0.5  # 底盘厚度
BASE_RADIUS = 0.02  # 底盘半径
WHEEL_THICKNESS = 0.1  # 轮子厚度
WHEEL_RADIUS = 0.2  # 轮子半径

"""
miniBox关节信息如下：
        joint index: 0,
        joint name:  b'base_to_front_follow_wheel_pillar',
        joint type:  0

        joint index: 1,
        joint name:  b'front_pillar_to_front_follow_wheel',
        joint type:  0

        joint index: 2,
        joint name:  b'joint_right_wheel',
        joint type:  0

        joint index: 3,
        joint name:  b'joint_left_wheel',
        joint type:  0
"""


# 获取节点信息
def getJointInfo(robot_id: int, physicsClientId: int = 0):
    for i in range(p.getNumJoints(robot_id, physicsClientId=physicsClientId)):
        info = p.getJointInfo(robot_id, i, physicsClientId=physicsClientId)
        print(f"""
            joint index: {info[0]},
            joint name:  {info[1]},
            joint type:  {info[2]}
        """)


# 根据输入的键盘字典执行小车运动
def control_miniBox(key_dict: dict, physicsClientId: int = 0):
    if len(key_dict) == 0:
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=[LEFT_WHEEL_JOINT_INDEX, RIGHT_WHEEL_JOINT_INDEX],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[0, 0],
            forces=[0, 0],
            physicsClientId=physicsClientId
        )
    # 先判断是否按下了上键和左键或者上键和右键，如果是，则让机器人拐弯左前，或者拐弯右前
    if UP in key_dict or LEFT in key_dict:
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=LEFT_WHEEL_JOINT_INDEX,
            controlMode=p.TORQUE_CONTROL,
            force=1.2
        )
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=RIGHT_WHEEL_JOINT_INDEX,
            controlMode=p.TORQUE_CONTROL,
            force=-1.2
        )
    #     p.setJointMotorControlArray(
    #             bodyUniqueId=robot_id,
    #             jointIndices=[LEFT_WHEEL_JOINT_INDEX, RIGHT_WHEEL_JOINT_INDEX],
    #             controlMode=p.TORQUE_CONTROL,
    #             # targetVelocities=[TARGET_VELOCITY / MULTIPLY, TARGET_VELOCITY],
    #             forces=[1.2, 1.2],
    #             physicsClientId=physicsClientId
    #     )
    # elif UP in key_dict and RIGHT in key_dict:
    #     p.setJointMotorControlArray(
    #             bodyUniqueId=robot_id,
    #             jointIndices=[LEFT_WHEEL_JOINT_INDEX, RIGHT_WHEEL_JOINT_INDEX],
    #             controlMode=p.TORQUE_CONTROL,
    #             # targetVelocities=[TARGET_VELOCITY, TARGET_VELOCITY / MULTIPLY],
    #             forces=[MAX_FORCE, MAX_FORCE],
    #             physicsClientId=physicsClientId
    #     )
    # elif DOWN in key_dict and LEFT in key_dict:
    #     p.setJointMotorControlArray(
    #             bodyUniqueId=robot_id,
    #             jointIndices=[LEFT_WHEEL_JOINT_INDEX, RIGHT_WHEEL_JOINT_INDEX],
    #             controlMode=p.TORQUE_CONTROL,
    #             # targetVelocities=[-1. * TARGET_VELOCITY, -1. * TARGET_VELOCITY / MULTIPLY],
    #             forces=[MAX_FORCE, MAX_FORCE],
    #             physicsClientId=physicsClientId
    #     )
    # elif DOWN in key_dict and RIGHT in key_dict:
    #     p.setJointMotorControlArray(
    #             bodyUniqueId=robot_id,
    #             jointIndices=[LEFT_WHEEL_JOINT_INDEX, RIGHT_WHEEL_JOINT_INDEX],
    #             controlMode=p.TORQUE_CONTROL,
    #             # targetVelocities=[-1. * TARGET_VELOCITY / MULTIPLY, -1. * TARGET_VELOCITY],
    #             forces=[MAX_FORCE, MAX_FORCE],
    #             physicsClientId=physicsClientId
    #     )
    # # 前后左右
    # elif UP in key_dict:
    #     p.setJointMotorControlArray(
    #             bodyUniqueId=robot_id,
    #             jointIndices=[LEFT_WHEEL_JOINT_INDEX, RIGHT_WHEEL_JOINT_INDEX],
    #             controlMode=p.TORQUE_CONTROL,
    #             # targetVelocities=[TARGET_VELOCITY, TARGET_VELOCITY],
    #             forces=[12, -12],
    #             physicsClientId=physicsClientId
    #     )
    #     # p.setJointMotorControl2(robot_id, LEFT_WHEEL_JOINT_INDEX, p.TORQUE_CONTROL, force=10)
    #     # p.setJointMotorControl2(robot_id, RIGHT_WHEEL_JOINT_INDEX, p.TORQUE_CONTROL, force=-10)
    # elif DOWN in key_dict:
    #     p.setJointMotorControlArray(
    #             bodyUniqueId=robot_id,
    #             jointIndices=[LEFT_WHEEL_JOINT_INDEX, RIGHT_WHEEL_JOINT_INDEX],
    #             controlMode=p.TORQUE_CONTROL,
    #             # targetVelocities=[-1. * TARGET_VELOCITY, -1. * TARGET_VELOCITY],
    #             forces=[MAX_FORCE, MAX_FORCE],
    #             physicsClientId=physicsClientId
    #     )
    # elif LEFT in key_dict:
    #     p.setJointMotorControlArray(
    #             bodyUniqueId=robot_id,
    #             jointIndices=[LEFT_WHEEL_JOINT_INDEX, RIGHT_WHEEL_JOINT_INDEX],
    #             controlMode=p.TORQUE_CONTROL,
    #             # targetVelocities=[-1. * TARGET_VELOCITY, TARGET_VELOCITY],
    #             forces=[MAX_FORCE, MAX_FORCE],
    #             physicsClientId=physicsClientId
    #     )
    # elif RIGHT in key_dict:
    #     p.setJointMotorControlArray(
    #             bodyUniqueId=robot_id,
    #             jointIndices=[LEFT_WHEEL_JOINT_INDEX, RIGHT_WHEEL_JOINT_INDEX],
    #             controlMode=p.TORQUE_CONTROL,
    #             # targetVelocities=[TARGET_VELOCITY, -1. * TARGET_VELOCITY],
    #             forces=[MAX_FORCE, MAX_FORCE],
    #             physicsClientId=physicsClientId
    #     )
    print(p.getJointState(robot_id, LEFT_WHEEL_JOINT_INDEX))
    print(p.getJointState(robot_id, RIGHT_WHEEL_JOINT_INDEX))


def setCameraPicAndGetPic(robot_id: int, width: int = 224, height: int = 224, physicsClientId: int = 0):
    """
    给合成摄像头设置图像并返回robot_id对应的图像
    摄像头的位置为miniBox前头的位置
    """
    basePos, baseOrientation = p.getBasePositionAndOrientation(robot_id, physicsClientId=physicsClientId)
    # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
    matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)
    tx_vec = np.array([matrix[0], matrix[3], matrix[6]])  # 变换后的x轴
    tz_vec = np.array([matrix[2], matrix[5], matrix[8]])  # 变换后的z轴

    basePos = np.array(basePos)
    # 摄像头的位置
    cameraPos = basePos + BASE_RADIUS * tx_vec + 0.5 * BASE_THICKNESS * tz_vec
    targetPos = cameraPos + 1 * tx_vec

    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=tz_vec,
        physicsClientId=physicsClientId
    )
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=50.0,  # 摄像头的视线夹角
        aspect=1.0,
        nearVal=0.01,  # 摄像头焦距下限
        farVal=20,  # 摄像头能看上限
        physicsClientId=physicsClientId
    )

    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=width, height=height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        physicsClientId=physicsClientId
    )

    return width, height, rgbImg, depthImg, segImg


# 添加门
def addDoor(a: list, b: list, c: list, d: list, color: list = [0., 0., 0.], width: int = 1, physicsClientId: int = 0):
    """
    a,b,c,d 代表门的四个角的坐标
    return: 勾勒出门的四条边的debug线的id
    """
    ab = p.addUserDebugLine(a, b, lineColorRGB=color, lineWidth=width, physicsClientId=physicsClientId)
    bc = p.addUserDebugLine(b, c, lineColorRGB=color, lineWidth=width, physicsClientId=physicsClientId)
    cd = p.addUserDebugLine(c, d, lineColorRGB=color, lineWidth=width, physicsClientId=physicsClientId)
    da = p.addUserDebugLine(d, a, lineColorRGB=color, lineWidth=width, physicsClientId=physicsClientId)
    return ab, bc, cd, da


"""
    以下的都是添加实体的几个函数，未说明的情况下，其中的pos都是形体与地面接触集合面的几何中心在世界坐标系中的坐标
    因此所有的pos参数的第三维正常情况下都是0，代表紧贴地面
"""


# 添加圆柱体实体
def addCylinder(pos: list, raidus: float, length: float, mass: float = 10000., rgba: list = [1., 1., 1., 1.],
                physicsClientId: int = 0):
    visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=raidus, length=length, rgbaColor=rgba,
                                       physicsClientId=physicsClientId)
    collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=raidus, height=length,
                                             physicsClientId=physicsClientId)
    entity_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[pos[0], pos[1], pos[2] + length / 2.],
        physicsClientId=physicsClientId
    )
    return entity_id


def addSphere(pos: list, radius: float, mass: float = 10000., rgba: list = [1., 1., 1., 1.], physicsClientId: int = 0):
    visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=physicsClientId)
    collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=physicsClientId)
    entity_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[pos[0], pos[1], pos[2] + radius],
        physicsClientId=physicsClientId
    )
    return entity_id


def addBox(pos: list, halfExtents: list, mass: float = 10000., rgba=[1., 1., 1., 1.], physicsClientId: int = 0):
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=rgba,
                                       physicsClientId=physicsClientId)
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents, physicsClientId=physicsClientId)
    entity_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[pos[0], pos[1], pos[2] + halfExtents[2]],
        physicsClientId=physicsClientId
    )


def addFence(center_pos: list, internal_length: float, internal_width: float, height: float, thickness: float,
             mass: float = 10000., rgba: list = [1., 1., 1., 1.], physicsClientId: int = 0):
    """
    :param center_pos:      围墙中心的坐标
    :param internal_length: 内部长
    :param internal_width:  内部宽
    :param thickness:       厚度
    :param mass:            质量
    :param rgba:            表面意思
    :return                 四个id，代表组成围墙的四个box的id
    """
    # L1和L2代表长那条线面对面的两面墙，长度为internal_length + 2 * thickness
    L1 = addBox(
        pos=[center_pos[0] + internal_width / 2. + thickness / 2., center_pos[1], center_pos[2]],
        halfExtents=[thickness / 2., internal_length / 2. + thickness, height / 2.],
        mass=mass / 4.,
        rgba=rgba,
        physicsClientId=physicsClientId
    )
    L2 = addBox(
        pos=[center_pos[0] - internal_width / 2. - thickness / 2., center_pos[1], center_pos[2]],
        halfExtents=[thickness / 2., internal_length / 2. + thickness, height / 2.],
        mass=mass / 4.,
        rgba=rgba,
        physicsClientId=physicsClientId
    )
    # W1和W2代表宽那条线面对面的两面墙，长度为internal_length + 2 * thickness
    W1 = addBox(
        pos=[center_pos[0], center_pos[1] + internal_length / 2. + thickness / 2., center_pos[2]],
        halfExtents=[internal_width / 2., thickness / 2., height / 2.],
        mass=mass / 4.,
        rgba=rgba,
        physicsClientId=physicsClientId
    )
    W2 = addBox(
        pos=[center_pos[0], center_pos[1] - internal_length / 2. - thickness / 2., center_pos[2]],
        halfExtents=[internal_width / 2., thickness / 2., height / 2.],
        mass=mass / 4.,
        rgba=rgba,
        physicsClientId=physicsClientId
    )
    return L1, L2, W1, W2


# 进行一组激光探测
def rayTest(robot_id: int, ray_length: float, ray_num: int = 5, physicsClientId: int = 0):
    """
    :param robot_id:   不多说
    :param ray_length: 激光长度
    :param ray_num:    激光数量(需要说明，激光头均匀分布在-pi/2到pi/2之间)
    """
    basePos, baseOrientation = p.getBasePositionAndOrientation(robot_id, physicsClientId=physicsClientId)

    # 将雷达与机器人对齐
    deg_to_rad = np.pi / 180.
    roll_angle = 270 * deg_to_rad
    roll_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(roll_angle), -np.sin(roll_angle)],
        [0, np.sin(roll_angle), np.cos(roll_angle)]
    ])

    # 创建表示沿pitch轴旋转90度的旋转矩阵
    yaw_angle = 90 * deg_to_rad
    yaw_matrix = np.array([
        [np.cos(yaw_angle), -np.sin(yaw_angle), 0],
        [np.sin(yaw_angle), np.cos(yaw_angle), 0],
        [0, 0, 1]
    ])
    matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)
    basePos = np.array(basePos)
    # basePos[2] = basePos[2] + BASE_T
    matrix = np.array(matrix).reshape([3, 3])
    matrix = matrix.dot(roll_matrix)
    matrix = matrix.dot(yaw_matrix)

    # 选定在机器人的本地坐标系中中心到几个激光发射点的向量
    # 此处的逻辑为先计算出local坐标系中的距离单位向量，再变换到世界坐标系中
    unitRayVecs = np.array([[cos(alpha), sin(alpha), 0] for alpha in np.linspace(-np.pi / 2., np.pi / 2., ray_num)])
    unitRayVecs = unitRayVecs.dot(matrix.T)
    # 通过广播运算得到世界坐标系中所有激光发射点的坐标
    rayBegins = basePos + BASE_RADIUS * unitRayVecs

    rayTos = rayBegins + ray_length * unitRayVecs
    results = p.rayTestBatch(rayBegins, rayTos, physicsClientId=physicsClientId)
    return rayBegins, rayTos, results


def checkCollision(robot_id: int, debug: bool, physicsClientId: int = 0):
    if p.getContactPoints(bodyA=robot_id, linkIndexA=-1, physicsClientId=physicsClientId):
        if debug:
            print("collsion happen!")
        return True
    # P_min, P_max = p.getAABB(robot_id)
    # id_tuple = p.getOverlappingObjects(P_min, P_max)
    # if len(id_tuple) > 1:
    #     for ID, _ in id_tuple:
    #         if ID == robot_id:      # 自己于自己的碰撞不算
    #             continue
    #         else:
    #             if debug:
    #                 print(f"hit happen! hit object is {p.getBodyInfo(ID)}")
    #             return True
    return False


if __name__ == "__main__":
    # 连接引擎
    cid = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # 添加资源路径
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # 载入机器人和其他的物件
    plane_id = p.loadURDF("plane.urdf")
    urdf_path = os.path.join(os.path.dirname(__file__), "urdf/balancebot.urdf")
    # urdf_path = os.path.join(os.path.dirname(__file__), "urdf/miniBox.urdf")
    robot_id = p.loadURDF(urdf_path, basePosition=ROBOT_POS,
                          baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2.]))
    ROBOT_POS, ROBOT_Orientation = p.getBasePositionAndOrientation(robot_id)
    ROBOT_Orientation = (0, 0.7071067811865475, 0.7071067811865475, 0)
    # 加入几个足球
    # p.loadURDF("soccerball.urdf", basePosition=[3, 3, 0], useMaximalCoordinates=True)
    # p.loadURDF("soccerball.urdf", basePosition=[-3, 3, 0], useMaximalCoordinates=True)
    # p.loadURDF("soccerball.urdf", basePosition=[3, -3, 0], useMaximalCoordinates=True)
    # p.loadURDF("soccerball.urdf", basePosition=[-3, -3, 0], useMaximalCoordinates=True)

    # 加入圆柱体实体
    addCylinder(pos=[2, 2, 0], raidus=0.5, length=2.)
    addSphere(pos=[-2, 2, 0], radius=1.)

    addFence(
        center_pos=[0, 0, 0],
        internal_length=20,
        internal_width=20,
        height=4,
        thickness=2,
        mass=10000.
    )

    # miniBox的两个随动关节设为禁用
    # p.setJointMotorControlArray(
    #     bodyUniqueId=robot_id,
    #     jointIndices=[0, 1],
    #     controlMode=p.VELOCITY_CONTROL,
    #     forces=[0., 0.]
    # )

    p.setGravity(0, 0, -9.8)
    # p.setRealTimeSimulation(1)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # 添加一个门，并增加显示和机器人距离的文本
    addDoor([-2, 1, 0], [-2, -1, 0], [-2, -1, 3], [-2, 1, 3], color=[0., 1., 0.], width=2.)
    door_id = p.addUserDebugText(
        text="",
        textPosition=DOOR_POS,
        textColorRGB=DEBUG_TEXT_COLOR,
        textSize=DEBUG_TEXT_SIZE
    )
    # 设置机器人位置和欧拉角的跟随文本
    text_id = p.addUserDebugText(
        text="",
        textPosition=[ROBOT_POS[0], ROBOT_POS[1], ROBOT_POS[2] + 1],
        textColorRGB=DEBUG_TEXT_COLOR,
        textSize=DEBUG_TEXT_SIZE
    )
    # 设置激光debug线
    # rayDebugLineIds = []
    # froms, tos, results = rayTest(robot_id, ray_length=5)
    # for index, result in enumerate(results):
    #     color = MISS_COLOR if result[0] == -1 else HIT_COLOR
    #     rayDebugLineIds.append(p.addUserDebugLine(froms[index], tos[index], color))

    # 增加控件，主要是重置用的reset按钮和调节全局参数的三个slider
    """
    slider控制的参数为：
            MAX_FORCE 差分驱动轮的最大马力                          (>0)
            TARGET_VELOCITY 差分驱动轮的最大速度                    (>0)
            MULTIPLY 转弯时，差分驱动轮的两个轮子的速度之比（大的比上小的） (>1)
    """
    MAX_FORCE_param_id = p.addUserDebugParameter("MAX_FORCE", 0, 100, MAX_FORCE)
    TARGET_VELOCITY_param_id = p.addUserDebugParameter("TARGET_VELOCITY", 0, 100, TARGET_VELOCITY)
    MULTIPLY_param_id = p.addUserDebugParameter("MULTIPLY", 1, 10, MULTIPLY)

    reset_btn_id = p.addUserDebugParameter("reset", 1, 0, 0)
    previous_btn_value = p.readUserDebugParameter(reset_btn_id)

    # 开始测试
    getJointInfo(robot_id)

    while True:
        # 获取三维坐标和欧拉角
        basePos, baseOrientation = p.getBasePositionAndOrientation(robot_id)
        baseEuler = p.getEulerFromQuaternion(baseOrientation)
        # p.setTimeStep(0.05)
        p.stepSimulation()
        time.sleep(1 / 240)
        # p.setRealTimeSimulation(1)
        # 读取并执行键盘信息
        key_dict = p.getKeyboardEvents()
        control_miniBox(key_dict)
        # 读取并更新slider中的参数
        MAX_FORCE = p.readUserDebugParameter(MAX_FORCE_param_id)
        TARGET_VELOCITY = p.readUserDebugParameter(TARGET_VELOCITY_param_id)
        MULTIPLY = p.readUserDebugParameter(MULTIPLY_param_id)
        # 判断按钮是否被按下，若是，则重置机器人的位置
        # TODO: reset机制有点问题，问题在于 resetBasePositionAndOrientation 函数只能重置base的位置，对于其余的link，会因扯回来的惯性而破坏平衡性
        # if p.readUserDebugParameter(reset_btn_id) != previous_btn_value:
        if SPACE_CLS in {key for key, value in key_dict.items() if value & p.KEY_WAS_TRIGGERED}:
            p.resetBasePositionAndOrientation(robot_id, ROBOT_POS, ROBOT_Orientation)
            previous_btn_value = p.readUserDebugParameter(reset_btn_id)

        # 更新debug文本
        text = f"Pos:   {[round(x, 2) for x in basePos]}\n                    Euler: {[round(x, 2) for x in baseEuler]}"
        text_id = p.addUserDebugText(
            text=text,
            textPosition=[basePos[0], basePos[1], basePos[2] + 1],

            textColorRGB=DEBUG_TEXT_COLOR,
            textSize=DEBUG_TEXT_SIZE,
            replaceItemUniqueId=text_id
        )
        # 计算离门的距离
        distance = np.linalg.norm(np.array(basePos) - np.array(DOOR_POS))
        text = f"Distance: {round(distance, 2)}"
        door_id = p.addUserDebugText(
            text=text,
            textPosition=[DOOR_POS[0], DOOR_POS[1], DOOR_POS[2] + 3],
            textColorRGB=DEBUG_TEXT_COLOR,
            textSize=DEBUG_TEXT_SIZE,
            replaceItemUniqueId=door_id
        )
        # 设置合成摄像头
        setCameraPicAndGetPic(robot_id)
        # 激光探测
        # froms, tos, results = rayTest(robot_id, ray_length=10.)
        # for index, result in enumerate(results):
        #     rayDebugLineIds[index] = p.addUserDebugLine(
        #         lineFromXYZ=froms[index], 
        #         lineToXYZ=tos[index] if result[0] == -1 else result[3], 
        #         lineColorRGB=MISS_COLOR if result[0] == -1 else HIT_COLOR, 
        #         lineWidth=RAY_DEBUG_LINE_WIDTH, 
        #         replaceItemUniqueId=rayDebugLineIds[index]
        #     )
        # 碰撞探测
        checkCollision(robot_id, debug=True)

    p.disconnect(cid)
