import gym
import numpy as np
from gym import spaces, envs
from envs.simulator import NetworkSimulatorEnv
from agents.q_agent import networkTabularQAgent


####
# This script currently makes the agent use the a random policy to explor the state,action space
# then tests the value of the learning by every 1000 iterations using the best choice and printing the reward
####

def main():
    callmean = 1.0
    for i in range(10):
        callmean += 1.0
        env = NetworkSimulatorEnv()
        state_pair = env._reset()
        env.callmean = callmean
        # agent 的初始化
        agent = networkTabularQAgent(env.nnodes, env.nedges, env.distance, env.nlinks)
        done = False
        r_sum_random = r_sum_best = 0
        config = agent.config

        for t in range(10001):
            if not done:

                current_state = state_pair[1]
                n = current_state[0]
                dest = current_state[1]

                # 更新q值，假装执行了每个action，但没有走（更新状态），只更新了q表值
                for action in range(env.nlinks[n]):
                    reward, next_state = env.pseudostep(action)
                    agent.learn(current_state, next_state, reward, action, done, env.nlinks)
                # 这里才是act寻找选取最佳action，并step执行了这个action，更新自身状态
                action = agent.act(current_state, env.nlinks)
                state_pair, reward, done, _ = env._step(action)

                next_state = state_pair[0]
                agent.learn(current_state, next_state, reward, action, done, env.nlinks)
                r_sum_random += reward

                if t % 10000 == 0:

                    if env.routed_packets != 0:
                        print(
                            "q learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_random:{}".format(
                                i, t, float(env.total_routing_time) / float(env.routed_packets),
                                float(env.total_hops) / float(env.routed_packets), r_sum_random))

                    # 更新state
                    current_state = state_pair[1]
                    n = current_state[0]
                    dest = current_state[1]

                    for action in range(env.nlinks[n]):
                        reward, next_state = env.pseudostep(action)
                        agent.learn(current_state, next_state, reward, action, done, env.nlinks)

                    action = agent.act(current_state, env.nlinks, True)
                    state_pair, reward, done, _ = env._step(action)

                    next_state = state_pair[0]
                    agent.learn(current_state, next_state, reward, action, done, env.nlinks)
                    r_sum_best += reward

                    if env.routed_packets != 0:
                        print
                        "q learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_best:{}".format(
                            i, t, float(env.total_routing_time) / float(env.routed_packets),
                            float(env.total_hops) / float(env.routed_packets), r_sum_best)


if __name__ == '__main__':
    main()
