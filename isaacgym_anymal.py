
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import time

args = get_args()
args.headless = True # For achieving maximum speed, all functions related to rendering are removed
env, _ = task_registry.make_env(name=args.task, args=args)

env.reset()

start_time = time.time()
for i in range(1000):
    # As we are using control_dofs_pos in Genesis, so I keep pd controls in legged_gym
    # Functions for fetching any kind of observation, domain randomization are removed
    # decimation = 1
    env.step(torch.empty(env.num_envs, 12).uniform_(-0.1, 0.1))
end_time = time.time()

print(env.num_envs)
print("Time:", end_time - start_time, "s")
print("#Step:", 1000 * env.num_envs)
print("FPS:", 1000 * env.num_envs / (end_time - start_time))
