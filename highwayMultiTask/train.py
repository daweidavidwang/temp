from tensorboardX import SummaryWriter

from dqnagent import DQNAgent
from multitask import multiTaskRL

EPISODES = 20000 

def main():
    mtr = multiTaskRL()
    dqn = DQNAgent(mtr.env)
    for episode in range(EPISODES):
        state = mtr.reset()
        while True:
            mtr.env.render()
            action = dqn.act(state)
            obs,done = mtr.step(action)

            record = mtr.getRecord() 
            if record is not None:
                pre_obs,action,cur_obs,reward,done,future_trajectory,info = \
                    mtr.deComposeRecord(record)
                dqn.record(pre_obs,action,reward,cur_obs,done,future_trajectory,info)
            
            if done:
                obs = mtr.reset()
            state = obs
            
if __name__ == "__main__":
    main()