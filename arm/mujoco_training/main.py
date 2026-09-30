import time
import mujoco
import mujoco.viewer
import numpy as np

# ---------------- 1. 加载模型 ----------------
model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

# ---------------- 2. 关节极限位置 ----------------
JOINT_LIMITS = [
    (-3.14, 3.14),   # J1 腰部回转
    (-1.57, 1.57),   # J2 肩部俯仰
    (-2.40, 2.40),   # J3 肘部俯仰
    (-1.57, 1.57),   # J4 腕部俯仰
    (-3.14, 3.14),   # J5 腕部滚转
]

# 每个关节的摆动周期（秒）—— 错开周期，动作层次分明
JOINT_PERIODS = [8.0, 3.0, 4.0, 3.5, 5.0]

# 每个关节的初相位（弧度）—— 错开相位
JOINT_PHASES = [0.0, 0.0, 1.0, 2.0, 0.5]


# ---------------- 3. 运动控制函数 ----------------
def move_joints_between_limits(t,
                               data,
                               limits=JOINT_LIMITS,
                               periods=JOINT_PERIODS,
                               phases=JOINT_PHASES,
                               shrink=0.95):
    """让每个关节在它的两个极限位置之间平滑来回运动。"""
    for i, (low, high) in enumerate(limits):
        lo = low * shrink
        hi = high * shrink
        mid = 0.5 * (lo + hi)
        amp = 0.5 * (hi - lo)
        target = mid + amp * np.sin(2.0 * np.pi * t / periods[i] + phases[i])
        data.ctrl[i] = target


# ---------------- 4. 主循环 ----------------
last_print = -1.0
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running():
        step_start = time.time()
        t = time.time() - start

        move_joints_between_limits(t, data)
        mujoco.mj_step(model, data)
        viewer.sync()

        # 每秒打印一次，方便确认关节是否跟得上
        if t - last_print >= 1.0:
            last_print = t
            c = data.ctrl[:5]
            q = data.qpos[:5]
            print(f"t={t:5.2f}  "
                  f"ctrl=[{c[0]:+.2f} {c[1]:+.2f} {c[2]:+.2f} {c[3]:+.2f} {c[4]:+.2f}]  "
                  f"qpos=[{q[0]:+.2f} {q[1]:+.2f} {q[2]:+.2f} {q[3]:+.2f} {q[4]:+.2f}]")

        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)