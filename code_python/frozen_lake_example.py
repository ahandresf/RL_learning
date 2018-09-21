#!usr/bin/env python
print("This is my first test in open gym")
import gym
import numpy as np
import tensorflow as tf

env = gym.make('FrozenLake-v0')

#Belleman equation for Q
def compute_value_function(policy,gamma=1.0):
    value_table=np.zeros(env.nS) #initiate the value table with zeros, array of the same dimension of number of states
    threshold=1e-10 #when the value function change less than this we go out
    while True:
        update_value_table = np.copy(value_table)
        print("test")
        for state in range(env.nS): #for each state in the enviroment states
            action = policy(state) #performe the action store in the policy for this specific state
            #Belleman equation for Q
            value_table[state]=sum(trans_prob * (reward_prob + gamma*updated_value_table[next_state])
                                   for trans_prob, next_state,reward_prob, _ in env.P[state][action])#for this specific state compute the probable reward for all possible actions
        if (np.sum((np.fabs(updated_value_table - value_table))) <= threshold):
            break
    return value_table

#This function extract the policy from the Q_table
def extract_policy(value_table, gamma):
    policy = np.zeros(env.observation_space.n) #create an array of 16 elements = number of elements in the observation space
    for state in range(env.action_space.n):
        Q_table=np.zeros(env.action_space.n) #Array of size of the action space for that state, can this change
        for action in range(env.action_space.n): #iterate among the posible actions in that state
            for next_sr in env.P[state][action]: #after taking an action, iterate over all the possible NEXT_state_reward
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state])) #call the value table definition, and with that it does get the Q table
            policy[state]=np.argmax(Q_table) #The max value in the Q table.
            return policy
