import pygame
import time
import random
import pandas as pd
import numpy as np


class Forest:
    _width = 100
    _height = 100

    def __init__(self, mapsize, alpha=0.5, beta=0.5, delta_beta=0.5):
        self._width = mapsize[0]
        self._height = mapsize[1]
        # self._df = self.create_map()
        self._state = self.create_map()
        self.alpha = alpha
        self.beta = beta
        self.delta_beta = delta_beta

    def create_map(self):
        data = np.zeros((self._width, self._height))
        return pd.DataFrame(data)
<<<<<<< HEAD

    def reset_map(self):
        self._df = self.create_map()
        self._ishealthy = self.create_map()

    def getState(self):
        return self._df
=======

    def reset_map(self):
        self._state = self.create_map()
        # self._ishealthy = self.create_map()

    def getState(self):
        return self._state
>>>>>>> 5fe6721684e15363c81ddb7860552bc4cb2f0bdc

    def getHealthState(self):
        return self._state

    def updatePoint(self, coor, value):
        if 0 <= value <= 1:
            self._state.loc[coor[0], coor[1]] = value

    def randomUpdate(self):
        x = random.randint(0, self._width)
        y = random.randint(0, self._height)
        v = random.random()
        self._state.loc[x, y] = v

    def getSize(self):
        return (self._width, self._height)

    def initFire(self):
        x = random.randint(0, self._width - 1)
        y = random.randint(0, self._height - 1)
<<<<<<< HEAD
        self._df.loc[x, y] -= 0.1
        self._ishealthy.loc[x, y] = 0

    def setState(self, x, y, v):
        curr = self._ishealthy.loc[x, y]
        self._ishealthy.loc[x, y] = v
=======
        self._state.loc[x, y] = 1
        print("Init fire")
        # self._ishealthy.loc[x,y] = 0

    def setState(self, x, y, v):
        curr = self._state.loc[x, y]
        self._state.loc[x, y] = v
>>>>>>> 5fe6721684e15363c81ddb7860552bc4cb2f0bdc
        return curr == 0

    #  action[i] = (action,pos) pos=(x,y)
    def stateUpdate(self, actions):
        relative_positions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        updated = False
        burnt = 0
        for i in range(self._width):
            for j in range(self._height):
                count = 0
                if self._state.loc[i, j] == 2:
                    continue
<<<<<<< HEAD
                updated = True
                prob = random.randint(0, 5)
                if prob > 3:
                    self._df.loc[i, j] -= 0.1

                if self._df.loc[i, j] <= 0:
                    self._ishealthy.loc[i, j] = 1
                    self._df.loc[i, j] = 0
                    burnt += 1
                for dr, dc in relative_positions:
                    nx = i + dr
                    ny = j + dc
                    if 0 <= nx < self._width and 0 <= ny < self._height:
                        prob = random.randint(0, 10)
                        if 3 <= self._df.loc[i, j] <= 7:
                            prob += 1
                        if (
                            prob > 7
                            and self._ishealthy.loc[nx, ny] == 1
                            and self._df.loc[nx, ny] > 0
                            and 0.2 <= self._df.loc[i, j] <= 0.8
                        ):
                            self._ishealthy.loc[nx, ny] = 0
                            self._df.loc[nx, ny] -= 0.01
=======
                if self._state.loc[i, j] == 1:
                    updated = True
                if self._state.loc[i, j] == 0:
                    for pos in relative_positions:
                        x = i + pos[0]
                        y = j + pos[1]
                        if (
                            0 <= x < self._width
                            and 0 <= y < self._height
                            and self._state.loc[x, y] == 1
                        ):
                            count += 1

                    # get random value between 0-1

                    rand_val = random.randint(0,100)
                    # rand > 1 - self.alpha**count
                    if rand_val/100 > (0.95):
                        self._state.loc[i, j] = 1
                        # updated = True

                rand_val = random.randint(0, 100)
                count_agents = 0
                for action, pos in actions:
                    if action[4] and pos[0] == i and pos[1] == j:
                        count_agents += 1
                if count_agents > 0:
                    if rand_val > (self.beta - self.delta_beta * count_agents):
                        print("Updated to burnt")
                        self._state.loc[i, j] = 2
                        burnt += 1
                        # updated = True
                else:
                    print("Tyring...",rand_val)
                    if rand_val/100 > 0.99:
                        print("Updated to burnt")
                        self._state.loc[i, j] = 2
                        burnt += 1
                        # updated = True

>>>>>>> 5fe6721684e15363c81ddb7860552bc4cb2f0bdc
        return updated, burnt

    def getColor(self, x, y):
        # color_dict = {
        #     0: (50, 42, 47),
        #     1: (205, 4, 2),
        #     2: (253, 17, 15),
        #     3: (253, 47, 45),
        #     4: (253, 77, 75),
        #     5: (254, 106, 105),
        #     6: (247, 186, 8),
        #     7: (168, 197, 69),
        #     8: (155, 213, 158),
        #     9: (115, 197, 119),
        #     10: (95, 188, 99),
        # }

        color_dict = {
            0: (95, 188, 99),
            1: (254, 106, 105),
            2: (25, 15, 12),
        }
        return color_dict[max(int(self._state.loc[x, y]), 0)]

    def getList(self):
<<<<<<< HEAD
        return self._df.values.tolist()

    def get_area(self, x, y, min_x, min_y, max_x, max_y):
        matrix = self._ishealthy.iloc[min_x : max_x + 1, min_y : max_y + 1]
        if ((x - min_x)) < 2:
            row = 2 - (x - min_x)
            col = max_y - min_y + 1
            row_up = np.ones((row, col)) * 2
            matrix = np.vstack((row_up, matrix))

        if (max_x - x) < 2:
            row = 2 - (max_x - x)
            col = max_y - min_y + 1
            row_down = np.ones((row, col)) * 2
            matrix = np.vstack((matrix, row_down))

        if (max_y - y) < 2:
            row = 5
            col = 2 - (max_y - y)
            col_right = np.ones((row, col)) * 2
            matrix = np.hstack((matrix, col_right))

        if (y - min_y) < 2:
            row = 5
            col = 2 - (y - min_y)
            col_left = np.ones((row, col)) * 2
            matrix = np.hstack((col_left, matrix))
        # Matrix with 5X5
        print(matrix)
        return self._ishealthy
=======
        return self._state.values.tolist()

    def get_area(self, x1, y1, x2, y2):
        return self._state

>>>>>>> 5fe6721684e15363c81ddb7860552bc4cb2f0bdc


class Simulator:
    _tick = 60

    def __init__(self, map, resolution=(1000, 1000), lineWidth=2):
        self.resolution = resolution
        self.lineWidth = lineWidth
        self.map = map
        self.mapSize = map.getSize()
        self.evaluate_dimensions()
        self.agentPos = (0, 0)

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

<<<<<<< HEAD
    def init_game(self, display=True):
        if display == False:
            self.screen = pygame.display.set_mode(self.resolution, flags=pygame.HIDDEN)
        else:
            self.screen = pygame.display.set_mode(self.resolution, flags=pygame.SHOWN)
=======
    def init_game(self):
        self.screen = pygame.display.set_mode(self.resolution)
>>>>>>> 5fe6721684e15363c81ddb7860552bc4cb2f0bdc
        self.clock = pygame.time.Clock()
        self.map.initFire()
        self.clock.tick(60)

    def reset_game(self):
        self.map.reset_map()
        self.agentPos = (0, 0)
        self.map.initFire()

    def draw_agent(self):
        x = self.convert_column_to_x(self.agentPos[0], self.square_width)
        y = self.convert_row_to_y(self.agentPos[1], self.square_height)
        geometry = (x + 10, y + 10, self.square_width - 10, self.square_height - 10)
        pygame.draw.rect(self.screen, (0, 0, 255), geometry)

    def update_agent_pos(self, action):
        new_pos = (
            self.agentPos[0] + action[0] - action[1],
            self.agentPos[1] - action[2] + action[3],
        )
        if 0 <= new_pos[0] < self.mapSize[0] and 0 <= new_pos[1] < self.mapSize[1]:
            self.agentPos = new_pos

    def draw_trees(self):
        for row in range(self.mapSize[0]):
            for column in range(self.mapSize[1]):
                x = self.convert_column_to_x(column, self.square_width)
                y = self.convert_row_to_y(row, self.square_height)
                geometry = (x, y, self.square_width, self.square_height)
                pygame.draw.rect(self.screen, self.map.getColor(row, column), geometry)

    def extinguish(self):
        rand = random.randint(0, 5)
        if rand > 1:
            return self.map.setState(self.agentPos[0], self.agentPos[1], 1)
        return False

    def get_agent_state(self):
        (x, y) = self.agentPos
        min_x = max(x - 2, 0)
        min_y = max(y - 2, 0)
<<<<<<< HEAD
        max_x = min(x + 2, self.mapSize[0] - 1)
        max_y = min(y + 2, self.mapSize[1] - 1)
        return self.map.get_area(x, y, min_x, min_y, max_x, max_y)
=======
        max_x = min(x + 2, self.mapSize[0])
        max_y = min(y + 2, self.mapSize[1])
        return self.map.get_area(min_x, max_x, min_y, max_y)
>>>>>>> 5fe6721684e15363c81ddb7860552bc4cb2f0bdc

    def play_step(self, action=None):
        reward = 0
        self.screen.fill((0, 0, 0))
        prob = random.randint(1, 5)
        if prob > 3:
<<<<<<< HEAD
            updated, burnt = self.map.stateUpdate()
=======
            updated, burnt = self.map.stateUpdate([(action, self.agentPos)])
>>>>>>> 5fe6721684e15363c81ddb7860552bc4cb2f0bdc
        else:
            updated = True
        self.draw_trees()
        self.update_agent_pos(action)
        if action[4] == 1:
            if self.extinguish():
                reward = 10
            else:
                reward = -1
        # reward-=burnt
        self.draw_agent()
        state = self.get_agent_state()
        pygame.display.flip()
        # x=True
        # while x:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                # if event.type == pygame.KEYDOWN:
                #     x=False

        return reward, state, updated == False
