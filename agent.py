import pygame
import time
import random
import pandas as pd
import numpy as np
from simulation import Simulator , Forest
from time import sleep
import os


# agent state , new_state , action 
# action [0,0,0,0,1] five tuple [up,down,left,right,extinguish]

class Agent():
    
    def __init__(self):
        pass
    def get_action(self):
        action = [0,0,0,0,0]
        rand = random.randint(0,4)
        action[rand]=1
        return action

def train():
    map = Forest((20,20))
    simulator = Simulator(map)
    agent = Agent()

    simulator.init_game()
    total_reward=0
    while True:
        action=agent.get_action()
        reward,state=simulator.play_step(action)
        total_reward+=reward
        print(state)
        
train()
