import pygame
import time
import random
import pandas as pd
import numpy as np

class Forest:
    _width = 100
    _height = 100

    def __init__(self, mapsize):
        self._width = mapsize[0]
        self._height = mapsize[1]
        self._df = self.create_map()
        self._ishealthy = self.create_map()

    def create_map(self):
        data = np.ones((self._width, self._height))
        return pd.DataFrame(data)
    
    def reset_map(self):
        self._df = self.create_map()
        self._ishealthy = self.create_map()


    def getState(self):
        return self._df
    
    def getHealthState(self):
        return self._df

    def updatePoint(self, coor, value):
        if 0 <= value <= 1:
            self._dp.loc[coor[0], coor[1]] = value

    def randomUpdate(self):
        x = random.randint(0, self._width)
        y = random.randint(0, self._height)
        v = random.random()
        self._df.loc[x, y] = v

    def getSize(self):
        return (self._width, self._height)

    def initFire(self):
        x = random.randint(0, self._width-1)
        y = random.randint(0, self._height-1)
        self._df.loc[x, y] -= 0.1
        self._ishealthy.loc[x,y] = 0

    def setState(self,x,y,v):
        curr = self._ishealthy.loc[x,y]
        self._ishealthy.loc[x,y] = v
        return curr==0

    def stateUpdate(self):
        relative_positions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        updated = False
        for i in range(self._width):
            for j in range(self._height):
                if self._ishealthy.loc[i, j] == 1:
                    continue
                updated = True
                prob = random.randint(0, 5)
                if prob > 3:
                    self._df.loc[i, j] -= 0.1
                
                if self._df.loc[i,j] <=0:
                    self._ishealthy.loc[i,j] = 1
                    self._df.loc[i,j] = 0
                for dr, dc in relative_positions:
                    nx = i + dr
                    ny = j + dc
                    if 0 <= nx < self._width and 0 <= ny < self._height:
                        prob = random.randint(0, 10)
                        if 3 <= self._df.loc[i, j] <= 7:
                            prob += 1
                        if (prob > 7 and self._ishealthy.loc[nx, ny] == 1 and 0.2 <= self._df.loc[i, j] <= 0.8):
                            self._ishealthy.loc[nx,ny] = 0
                            self._df.loc[nx, ny] -= 0.01
        return updated

    def getColor(self, x, y):
        color_dict = {
            0: (50, 42, 47),
            1: (205, 4, 2),
            2: (253, 17, 15),
            3: (253, 47, 45),
            4: (253, 77, 75),
            5: (254, 106, 105),
            6: (247, 186, 8),
            7: (168, 197, 69),
            8: (155, 213, 158),
            9: (115, 197, 119),
            10: (95, 188, 99),
        }
        return color_dict[max(int(self._df.loc[x, y] * 10), 0)]

    def getList(self):
        return self._df.values.tolist()
    
    def get_area(self,x1,y1,x2,y2):
        return self._ishealthy

class Simulator:
    _tick = 1

    def __init__(self, map, resolution=(1000, 1000), lineWidth=2):
        self.resolution = resolution
        self.lineWidth = lineWidth
        self.map = map
        self.mapSize = map.getSize()
        self.evaluate_dimensions()
        self.agentPos = (0,0)

    def configure(self, tick):
        self._tick = tick

    def evaluate_dimensions(self):
        self.square_width = (self.resolution[0] / self.mapSize[0]) - self.lineWidth * (
            (self.mapSize[0] + 1) / self.mapSize[0]
        )
        self.square_height = (self.resolution[1] / self.mapSize[1]) - self.lineWidth * (
            (self.mapSize[1] + 1) / self.mapSize[1]
        )

    def convert_column_to_x(self, column, square_width):
        return self.lineWidth * (column + 1) + square_width * column

    def convert_row_to_y(self, row, square_height):
        return self.lineWidth * (row + 1) + square_height * row
    
    def init_game(self):
        self.screen = pygame.display.set_mode(self.resolution)
        self.clock = pygame.time.Clock()
        self.map.initFire()
        self.clock.tick(1)

    def reset_game(self):
        self.map.reset_map()
        self.agentPos = (0,0)
        self.map.initFire()

    def draw_agent(self):
        x = self.convert_column_to_x(self.agentPos[0], self.square_width)
        y = self.convert_row_to_y(self.agentPos[1], self.square_height)
        geometry = (x+10, y+10, self.square_width-10, self.square_height-10)
        pygame.draw.rect(self.screen,(0,0,255), geometry)

    def update_agent_pos(self,action):
        new_pos = (self.agentPos[0]+action[0]-action[1],self.agentPos[1]-action[2]+action[3])
        if 0<=new_pos[0]<self.mapSize[0] and 0<=new_pos[1]<self.mapSize[1]:
            self.agentPos = new_pos


    def draw_trees(self):
        for row in range(self.mapSize[0]):
            for column in range(self.mapSize[1]):
                x = self.convert_column_to_x(column, self.square_width)
                y = self.convert_row_to_y(row, self.square_height)
                geometry = (x, y, self.square_width, self.square_height)
                pygame.draw.rect(self.screen, self.map.getColor(row, column), geometry)
    
    def extinguish(self):
        rand = random.randint(0,5)
        if rand>1:
            return self.map.setState(self.agentPos[0],self.agentPos[1],1)
        return False
    
    def get_agent_state(self):
        (x,y) = self.agentPos
        min_x = max(x-2,0)
        min_y = max(y-2,0)
        max_x = min(x+2,self.mapSize[0])
        max_y = min(y+2,self.mapSize[1])
        return self.map.get_area(min_x,max_x,min_y,max_y)
    
    def play_step(self,action=None):
        reward=0
        self.screen.fill((0, 0, 0))
        updated = self.map.stateUpdate()
        self.draw_trees()
        self.update_agent_pos(action)
        if action[4]==1:
            if self.extinguish():
                reward=10
            else:
                reward=-5
        self.draw_agent()
        state = self.get_agent_state()
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                quit()
        return reward,state, updated==False
