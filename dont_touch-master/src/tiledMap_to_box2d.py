import pygame
import json
from .env import PPM, HEIGHT


# Map 讀取地圖資料
class TiledMap_box2d:
    def __init__(self, filename: str, TILESIZE):
        pygame.init()
        self.data = []
        with open(filename) as json_file:
            data = json.load(json_file)
            self.tileHeight = data["height"]
            self.tileWidth = data["width"]
            map_data = data["layers"][0]["data"]
            map = []
            # for i in range(self.tileHeight):
            #     self.data.append(data["layers"][0]["data"][i * self.tileWidth:(i + 1) * self.tileWidth])
            for i in range(len(map_data)):
                if map_data[i]:
                    self.data.append([i // self.tileWidth, i % self.tileWidth, map_data[i]])
        self.width = self.tileWidth * TILESIZE
        self.height = self.tileHeight * TILESIZE
        self.PPM = 20
        self.screen = pygame

    def get_wall_info(self):  # 擷取迷宮
        wall_tiles = []
        passed = []
        j = 0
        first_tile = -1
        last_tile = -1  # 前一顆tile
        type = ''
        while len(passed) < len(self.data):
            if j in passed:
                j += 1
                if j == len(self.data):
                    j = 0
                continue
            if self.data[j][2] == 1:
                if first_tile == -1:  # a new wall
                    first_tile = j  # index
                    last_tile = j
                    passed.append(j)
                    j += 1
                    if (self.data[j][0] == self.data[last_tile][0] and
                            self.data[j][1] == self.data[last_tile][1] + 1 and self.data[j][2] == 1):  # 檢查橫向有沒有延伸的積木
                        type = 'h'
                    else:
                        type = 'v'
                    continue
                elif type == 'h':  # 橫向積木
                    if (self.data[j][0] == self.data[last_tile][0] and
                            self.data[j][1] == self.data[last_tile][1] + 1):
                        last_tile = j
                        passed.append(j)
                        if j == len(self.data) - 1:
                            wall_tiles.append([self.data[first_tile], self.data[last_tile]])
                            j = 0
                            first_tile = -1
                            continue
                        else:
                            j += 1
                            continue
                    else:
                        wall_tiles.append([self.data[first_tile], self.data[last_tile]])
                        j = 0
                        first_tile = -1
                elif type == 'v':  # 縱向積木
                    if (self.data[j][1] == self.data[last_tile][1] and
                            self.data[j][0] == self.data[last_tile][0] + 1):
                        last_tile = j
                        passed.append(j)
                        if j == len(self.data) - 1:
                            wall_tiles.append([self.data[first_tile], self.data[last_tile]])
                            j = 0
                            first_tile = -1
                            continue
                        else:
                            j += 1
                            continue
                    elif (self.data[j][0] > self.data[last_tile][0] + 1 and self.data[j][1] > self.data[last_tile][1]):
                        wall_tiles.append([self.data[first_tile], self.data[last_tile]])
                        j = 0
                        first_tile = -1
                    else:
                        j += 1
                        if j == len(self.data) - 1:
                            wall_tiles.append([self.data[first_tile], self.data[last_tile]])
                            j = 0
                            first_tile = -1
                            continue
                        continue
                else:
                    continue
            else:
                passed.append(j)
                j += 1
                continue
        return wall_tiles

    def transfer_to_box2d(self, wall: list):
        '''
        :param wall:[[0 ,0 , 1],[0, 29, 1]]
        :return: vertices of box2d body
        (y, height - 1 - x) = box2d中左下角的點的座標
        '''
        vertices = []
        if wall[0][0] == wall[1][0]:  # 橫向積木
            left_tile_x = wall[0][1]
            left_tile_y = self.tileHeight - 1 - wall[0][0]
            right_tile_x = wall[1][1]
            right_tile_y = self.tileHeight - 1 - wall[1][0]
            vertices.extend([[left_tile_x, left_tile_y], [left_tile_x, left_tile_y + 1],
                             [right_tile_x + 1, right_tile_y], [right_tile_x + 1, right_tile_y + 1]])
        else:  # 縱向積木
            upper_tile_x = wall[0][1]
            upper_tile_y = self.tileHeight - 1 - wall[0][0]
            down_tile_x = wall[1][1]
            down_tile_y = self.tileHeight - 1 - wall[1][0]
            vertices.extend([[upper_tile_x, upper_tile_y + 1], [upper_tile_x + 1, upper_tile_y + 1],
                             [down_tile_x, down_tile_y], [down_tile_x + 1, down_tile_y]])
        return vertices

    def load_other_obj(self):
        obj = {"car": None,
               "end_point": None,
               "check_point": []}
        for i in range(len(self.data)):
            if self.data[i][2] == 1:
                continue
            elif self.data[i][2] in range(2, 6):
                # obj["car"] = self.data[i]
                obj["car"] = [self.tileHeight - 1 - self.data[i][0], self.data[i][1], self.data[i][2]]
            elif self.data[i][2] == 6:
                obj["end_point"] = [self.tileHeight - 1 - self.data[i][0], self.data[i][1], self.data[i][2]]
                # obj["end_point"] = self.data[i]
            elif self.data[i][2] == 7:
                obj["check_point"].append([self.tileHeight - 1 - self.data[i][0], self.data[i][1], self.data[i][2]])
                # obj["check_point"].append(self.data[i])
        return obj

    def get_check_wall_info(self):
        end = []
        for i in self.data:
            if i[2] == 7:
                end.append(self.data.index(i))

        wall_tiles = []
        passed = []
        j = 0
        first_tile = -1
        last_tile = -1  # 前一顆tile
        type = ''
        while len(passed) < len(self.data):
            if j in passed:
                j += 1
                if j == len(self.data):
                    j = 0
                continue
            if self.data[j][2] == 7:
                if first_tile == -1:  # a new wall
                    first_tile = j  # index
                    last_tile = j
                    passed.append(j)
                    end.remove(j)
                    j += 1
                    if (self.data[j][0] == self.data[last_tile][0] and
                            self.data[j][1] == self.data[last_tile][1] + 1 and
                            self.data[j][2] == 7):  # 檢查橫向有沒有延伸的積木
                        type = 'h'
                    else:
                        type = 'v'
                    continue
                elif type == 'h':  # 橫向積木
                    if (self.data[j][0] == self.data[last_tile][0] and
                            self.data[j][1] == self.data[last_tile][1] + 1):
                        last_tile = j
                        passed.append(j)
                        if j == end[-1]:
                            wall_tiles.append([self.data[first_tile], self.data[last_tile]])
                            end.remove(j)
                            j = 0
                            first_tile = -1
                            continue
                        else:
                            end.remove(j)
                            j += 1
                            continue
                    else:
                        wall_tiles.append([self.data[first_tile], self.data[last_tile]])
                        j = 0
                        first_tile = -1
                elif type == 'v':  # 縱向積木
                    if (self.data[j][1] == self.data[last_tile][1] and
                            self.data[j][0] == self.data[last_tile][0] + 1):
                        last_tile = j
                        passed.append(j)
                        if j == end[-1]:
                            wall_tiles.append([self.data[first_tile], self.data[last_tile]])
                            end.remove(j)
                            j = 0
                            first_tile = -1
                            continue
                        else:
                            end.remove(j)
                            j += 1
                            continue
                    elif (self.data[j][0] > self.data[last_tile][0] + 1):
                        wall_tiles.append([self.data[first_tile], self.data[last_tile]])
                        j = 0
                        first_tile = -1
                    elif j == end[-1]:
                        wall_tiles.append([self.data[first_tile], self.data[last_tile]])
                        j = 0
                        first_tile = -1
                        continue
                    else:
                        j += 1
                        if j == end[-1]:
                            wall_tiles.append([self.data[first_tile], self.data[last_tile]])
                            end.remove(j)
                            j = 0
                            first_tile = -1
                            continue
                        continue
                else:
                    continue
            else:
                passed.append(j)
                if j == len(self.data) - 1:
                    j = 0
                    continue
                j += 1
                continue
        return wall_tiles

    def print_data(self):
        print("width", self.width)
        print("height", self.height)
        print("tile_width", self.tileWidth)
        print("tile_height", self.tileHeight)
        for i in range(len(self.data)):
            print(self.data[i])
