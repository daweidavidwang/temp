import gym
import highway_env
import numpy as np

# config = {
#     "observation": {
#         "type": "Kinematics",
#         "vehicles_count": 5,
#         "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
#         "order": "sorted"
#     },
#     "absolute": True,
#     "normalized":False
# }
# env = gym.make('highway-v0')
# print(env.action_space)
# #env.configure(config)
# obs = env.reset()
# for _ in range(50):
#     action = env.action_type.actions_indexes["IDLE"]
#     obs, reward, done, info = env.step(action)
#     env.render()
#     print(type(obs[0][1:3]))
#     a,b = obs.shape
#     print(a,b)

a = np.array([[12,3,4,5],[24,5,6,32]])
a = a / 12
print(a)