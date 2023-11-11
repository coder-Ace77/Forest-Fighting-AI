import pygame
import time
import random
import pandas as pd
import torch
import numpy as np
from simulation import Simulator , Forest
from time import sleep
from collections import deque
import os
from helper import plot
from model import Linear_QNet,QTrainer

MAX_MEMORY = 100_000
LR = 0.0001
DSFACTOR = 0.9
BATCH_SIZE = 1000

# agent state , new_state , action 
# action [0,0,0,0,1] five tuple [up,down,left,right,extinguish]

class Agent():
    
    def __init__(self):

        self.memory=deque(maxlen=MAX_MEMORY)
        self.model=Linear_QNet(400, 512, 5)
        self.trainer = QTrainer(self.model, lr=LR, gamma=DSFACTOR)
    
    def convert_state(self,state):
        return state.values.reshape(-1,).tolist()
    
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((self.convert_state(state),action,reward,self.convert_state(next_state),done))

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, done = zip(*mini_sample)
        self.trainer.long_train_step(states, actions, rewards, next_states, done)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(self.convert_state(state), action, reward, self.convert_state(next_state), done)
    
    def get_action(self,state,gameIter):

        action = [0,0,0,0,0]

        x = 50 - gameIter
        if random.randint(0,100) < x:
            move = random.randint(0,4)
            # print("Random Move:",move)
            action[move]=1

        else:
            state0 = torch.tensor(self.convert_state(state),dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            # print("Pred Move:",move)
            action[move]=1
        # print(action)
        return action


def train():

    map = Forest((20,20))
    simulator = Simulator(map)
    agent = Agent()
    simulator.init_game()
    state = map.getHealthState()
    total_reward=0
    gameIter = 1

    rewards = []
    plot_scores = []
    plot_mean_scores = []
    total_score = 0

    while True:

        action=agent.get_action(state,gameIter)
        reward,next_state,done=simulator.play_step(action)

        agent.remember(state,action,reward,next_state,done)
        
        # train short memory
        agent.train_short_memory(state,action,reward,next_state,done)
        state = next_state

        total_reward+=reward
        if done:
            agent.train_long_memory()
            simulator.reset_game()
            # print("Reset game")
            # os.system('clear')
            rewards.append(total_reward)
            # print(rewards)
            sleep(1)
            total_reward=0
            # mean_score = total_score / gameIter
            # plot_scores.append(total_reward)
            # total_score += total_reward
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)
            gameIter+=1
train()
