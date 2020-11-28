import gym
import highway_env

env = gym.make('highway-v0')
print(env.action_space)
#env.configure(config)
obs = env.reset()
for _ in range(50):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, info = env.step(action)
    env.render()
    print(obs)