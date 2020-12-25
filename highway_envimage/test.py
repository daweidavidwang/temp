import gym
import highway_env
from matplotlib import pyplot as plt
from preprocess import ZFilter

env = gym.make("highway-v0")
running_state = ZFilter((84,84,5), clip=1.0)
screen_width, screen_height = 84, 84

config = {
    "offscreen_rendering": False,
    "observation": {
        "type": "GrayscaleObservation",
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "stack_size": 5,
        "observation_shape": (screen_width, screen_height)
    },
    "screen_width": screen_width,
    "screen_height": screen_height,
    "scaling": 1.75,
    "policy_frequency": 5
}
env.configure(config)
print(env.observation_space)
obs = env.reset()
print(env.observation_space)

done = False
while not done:
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, info = env.step(action)
    env.render()
    _, axes = plt.subplots(ncols=4, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(obs[..., i], cmap=plt.get_cmap('gray'))
    plt.show()


