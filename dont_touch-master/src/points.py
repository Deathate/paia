import pygame
from mlgame.game.paia_game import GameResultState, GameStatus
from mlgame.view.view_model import create_polygon_view_data
from .env import *


class Point(pygame.sprite.Sprite):
    def __init__(self, game, coordinate):
        self.group = game.all_points
        pygame.sprite.Sprite.__init__(self, self.group)
        self.game = game
        # self.x, self.y = (coordinate[0] - TILESIZE / PPM, -coordinate[1] + TILESIZE / PPM)
        self.x, self.y = (coordinate[0], coordinate[1])

    def get_info(self):
        return {
            "coordinate": (self.x, self.y)
        }

    def get_progress_data(self):
        asset_data = {}
        return asset_data


class End_point(Point):
    def __init__(self, game, coordinate):
        Point.__init__(self, game, coordinate)
        self.rect = pygame.Rect(self.x, self.y, TILESIZE * 3, TILESIZE * 3)

    def update(self, *args, **kwargs) -> None:
        self.detect_cars_collision()

    def detect_cars_collision(self):
        hits = pygame.sprite.spritecollide(self, self.game.cars, False)
        for hit in hits:
            if hit.is_running:
                hit.check_point += 1
                hit.end_frame = self.game.frame
                hit.is_completed = True
                self.game.eliminated_user.append(hit)  # TODO #外部注入
                self.game.state = GameResultState.FINISH
                hit.is_running = False
                hit.status = GameStatus.GAME_PASS

    def get_progress_data(self):
        asset_data = {"type": "image",
                      "x": self.rect.x,
                      "y": self.rect.y,
                      "width": self.rect.width,
                      "height": self.rect.height,
                      "image_id": "logo",
                      "angle": 0}
        return asset_data


class Check_point(Point):
    def __init__(self, game, vertices):
        Point.__init__(self, game, vertices[0])
        self.rect = pygame.Rect(vertices[0][0], vertices[0][1], abs(vertices[2][0] - vertices[1][0]), abs(vertices[2][1]-vertices[1][1]))
        self.car_has_hit = []
        self.vertices = vertices
        self.color = YELLOW

    def update(self, *args, **kwargs) -> None:
        self.color = YELLOW
        self.detect_cars_collision()

    def detect_cars_collision(self):
        hits = pygame.sprite.spritecollide(self, self.game.cars, False)
        for hit in hits:
            self.color = GREEN
            if hit.status and hit not in self.car_has_hit:
                hit.check_point += 1
                hit.end_frame = self.game.frame
                self.car_has_hit.append(hit)

    def get_progress_data(self):
        # asset_data = {"type": "rect",
        #               "name": 'check_point',
        #               "x": self.rect.x,
        #               "y": self.rect.y,
        #               "angle": 0,
        #               "width": self.rect.width,
        #               "height": self.rect.height,
        #               "color": YELLOW}
        asset_data = create_polygon_view_data("check_point", [list(v) for v in self.vertices], self.color)
        return asset_data


class Outside_point(Point):
    '''
    if car colliding these point, car will be swich to start point
    '''

    def __init__(self, game, coordinate):
        Point.__init__(self, game, coordinate)
        # self.image = pygame.Surface((TILESIZE, TILESIZE))
        # self.image.fill(BLUE)
        # self.rect = self.image.get_rect()
        self.rect = pygame.Rect(self.x, self.y, TILESIZE * 3, TILESIZE * 3)

    def update(self, *args, **kwargs) -> None:
        self.detect_cars_collision()

    def detect_cars_collision(self):
        hits = pygame.sprite.spritecollide(self, self.game.cars, False)
        for hit in hits:
            if hit.status:
                hit.body.position = (hit.x, hit.y)
                hit.body.linearVelocity = 0, 0
                pass
