import numpy as np
import genesis as gs
import time

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################

scene = gs.Scene(
    show_viewer=False,
    rigid_options=gs.options.RigidOptions(
        dt=0.005,
        constraint_solver=gs.constraint_solver.Newton,
        enable_self_collision=True,
    ),
)

########################## entities ##########################
scene.add_entity(
    gs.morphs.Plane(),
)
LEGGED_GYM_ROOT_DIR = "/home/ziyanx/python/legged_gym/legged_gym"
robot = scene.add_entity(
    gs.morphs.URDF(
        file=f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt",
        pos=(0, 0, 0.8),
    ),
)
########################## build ##########################
n_envs = 30000
scene.build(n_envs=n_envs)

joint_names = [
    "RH_HAA",
    "LH_HAA",
    "RF_HAA",
    "LF_HAA",
    "RH_HFE",
    "LH_HFE",
    "RF_HFE",
    "LF_HFE",
    "RH_KFE",
    "LH_KFE",
    "RF_KFE",
    "LF_KFE",
]
motor_dofs = [robot.get_joint(name).dof_idx_local for name in joint_names]

robot.set_dofs_kp(np.full(12, 80), motor_dofs)
robot.set_dofs_kv(np.full(12, 2), motor_dofs)
robot.control_dofs_position(np.zeros((n_envs, 12)), motor_dofs)

start_time = time.time()
for i in range(1000):
    robot.control_dofs_position(np.random.uniform(-0.1, 0.1, size=(n_envs, 12)), motor_dofs)
    scene.step()
end_time = time.time()

print(n_envs)
print("Time:", end_time - start_time, "s")
print("#Step:", 1000 * n_envs)
print("FPS:", 1000 * n_envs / (end_time - start_time))
