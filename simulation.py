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

    def create_map(self):
        data = np.ones((self._width, self._height))
        return pd.DataFrame(data)

    def getState(self):
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
        x = random.randint(0, self._width)
        y = random.randint(0, self._height)
        self._df.loc[x, y] -= 0.1

    def stateUpdate(self):
        relative_positions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for i in range(self._width):
            for j in range(self._height):
                if self._df.loc[i, j] == 1 or self._df.loc[i, j] <= 0:
                    continue
                prob = random.randint(0, 5)
                if prob > 3:
                    self._df.loc[i, j] -= 0.1
                for dr, dc in relative_positions:
                    nx = i + dr
                    ny = j + dc
                    if 0 <= nx < self._width and 0 <= ny < self._height:
                        prob = random.randint(0, 10)
                        if 3 <= self._df.loc[i, j] <= 7:
                            prob += 1
                        if (
                            prob > 7
                            and self._df.loc[nx, ny] == 1
                            and 0.2 <= self._df.loc[i, j] <= 0.8
                        ):
                            self._df.loc[nx, ny] -= 0.01

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
        # print(x, y)
        return color_dict[max(int(self._df.loc[x, y] * 10), 0)]

    def getList(self):
        return self._df.values.tolist()


class Simulator:
    _tick = 1

    def __init__(self, mapSize, resolution=(1000, 1000), lineWidth=2):
        self.resolution = resolution
        self.lineWidth = lineWidth
        self.mapSize = mapSize
        self.evaluate_dimensions()

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

    def draw_squares(self, map):
        map_size = map.getSize()
        for row in range(map_size[0]):
            for column in range(map_size[1]):
                x = self.convert_column_to_x(column, self.square_width)
                y = self.convert_row_to_y(row, self.square_height)
                geometry = (x, y, self.square_width, self.square_height)
                pygame.draw.rect(self.screen, map.getColor(row, column), geometry)

    def run(self, map):
        self.screen = pygame.display.set_mode(self.resolution)
        self.clock = pygame.time.Clock()
        map.initFire()
        while True:
            self.clock.tick(1)
            self.screen.fill((0, 0, 0))
            self.draw_squares(map)
            pygame.display.flip()
            map.stateUpdate()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()


mapsize = (20, 20)

forest = Forest(mapsize)
simulator = Simulator(mapsize, (1000, 1000))
simulator.run(forest)
