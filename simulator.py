import pygame
import pandas as pd
import numpy as np


class Simulator:
    def __init__(self, size, agentNo, resolution=(1000, 1000), lineWidth=2):
        self.resolution = resolution
        self.lineWidth = lineWidth
        self.size = size
        self.agentNo = agentNo
        self.screen = pygame.display.set_mode(self.resolution)
        self.clock = pygame.time.Clock()
        self.clock.tick(1)
        self.evaluate_dimensions()

    def getColor(self, color):
        color_dict = {
            0: (0, 255, 0),
            1: (255, 0, 0),
            2: (0, 0, 0),
        }
        return color_dict[color]

    def evaluate_dimensions(self):
        self.square_width = (self.resolution[0] / self.size[0]) - self.lineWidth * (
            (self.size[0] + 1) / self.size[0]
        )
        self.square_height = (self.resolution[1] / self.size[1]) - self.lineWidth * (
            (self.size[1] + 1) / self.size[1]
        )

    def convert_column_to_x(self, column, square_width):
        return self.lineWidth * (column + 1) + square_width * column

    def convert_row_to_y(self, row, square_height):
        return self.lineWidth * (row + 1) + square_height * row

    def draw_agent(self, agentList):
        # print("INSIDE DRAW", agentList)
        for agent in agentList:
            x = self.convert_column_to_x(agent[0], self.square_width)
            y = self.convert_row_to_y(agent[1], self.square_height)
            geometry = (x + 1, y + 1, self.square_width - 1, self.square_height - 1)
            pygame.draw.rect(self.screen, (255, 255, 255), geometry)
        pygame.display.flip()

    def draw_trees(self, map):
        self.screen.fill((0, 0, 0))
        for row in range(self.size[0]):
            for column in range(self.size[1]):
                x = self.convert_column_to_x(column, self.square_width)
                y = self.convert_row_to_y(row, self.square_height)
                geometry = (x, y, self.square_width, self.square_height)
                pygame.draw.rect(self.screen, self.getColor(map[row][column]), geometry)
        pygame.display.flip()

    def quit(self):
        pygame.quit()
