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
        self.model=Linear_QNet(400, 512, 32 , 5)
        self.trainer = QTrainer(self.model, lr=LR, gamma=DSFACTOR)
    
    def convert_state(self,state):

        return np.array(state,dtype=int)
    
    def remember(self,state,action,next_state,done):

        self.memory.append((state,action,next_state,done))

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self,state,gameIter):

        action = [0,0,0,0,0]

        x = 50 - gameIter
        if random.randint(0,100) < x:
            move = random.randint(0,4)
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        action[move]=1
        
        return action




def train():

    map = Forest((20,20))
    simulator = Simulator(map)
    agent = Agent()
    simulator.init_game()
    state = map.getHealthState()
    total_reward=0
    gameIter = 1

    while True:

        action=agent.get_action(state,gameIter)
        reward,next_state,done=simulator.play_step(action)

        # remember
        agent.remember(state,action,next_state,done)
        state = next_state

        total_reward+=reward
        if done:
            simulator.reset_game()
            print(total_reward)
            total_reward=0
            gameIter+=1

if __name__=='__name__':
    train()
