import gym
import highway_env
from matplotlib import pyplot as plt
from preprocess import ZFilter

env = gym.make("highway-v0")
print(env.observation_space)
obs, pastpos, futurepos = env.reset()
print(env.observation_space)

done = False
while not done:
    action = env.action_type.actions_indexes["IDLE"]
    obs, pastpos, futurepos, reward, done, info = env.step(action)
    print(obs.shape)
    env.render()
    # _, axes = plt.subplots(ncols=4, figsize=(12, 5))
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(obs[..., i], cmap=plt.get_cmap('gray'))
    # plt.show()

