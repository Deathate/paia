import pygame
from os import path
from mlgame.view.view_model import create_asset_init_data
from .env import WINDOW_HEIGHT, WINDOW_WIDTH, IMAGE_DIR
from .GameFramework.Bullet import Bullet

vec = pygame.math.Vector2


class TankBullet(Bullet):
    def __init__(self, _id: int, center: tuple, width: int, height: int, rot: int):
        super().__init__(center, width, height)
        self.map_width = WINDOW_WIDTH
        self.map_height = WINDOW_HEIGHT
        self._id = _id
        self.rot = rot
        self.angle = 3.14 / 180 * (self.rot + 90)
        self.move = {"left_up": vec(-self.speed, -self.speed), "right_up": vec(self.speed, -self.speed),
                     "left_down": vec(-self.speed, self.speed), "right_down": vec(self.speed, self.speed),
                     "left": vec(-self.speed, 0), "right": vec(self.speed, 0), "up": vec(0, -self.speed),
                     "down": vec(0, self.speed)}

    def update_bullet(self):
        if self.rot == 0:
            self.rect.center += self.move["left"]
        elif self.rot == 315 or self.rot == -45:
            self.rect.center += self.move["left_up"]
        elif self.rot == 270 or self.rot == -90:
            self.rect.center += self.move["up"]
        elif self.rot == 225 or self.rot == -135:
            self.rect.center += self.move["right_up"]
        elif self.rot == 180 or self.rot == -180:
            self.rect.center += self.move["right"]
        elif self.rot == 135 or self.rot == -225:
            self.rect.center += self.move["right_down"]
        elif self.rot == 90 or self.rot == -270:
            self.rect.center += self.move["down"]
        elif self.rot == 45 or self.rot == -315:
            self.rect.center += self.move["left_down"]

    def get_image_init_data(self):
        img_data = {"bullet": "https://raw.githubusercontent.com/Jesse-Jumbo/TankMan/main/asset/image/bullet.png"}
        id, url = img_data.items()
        image_init_data = create_asset_init_data(id, self.rect.width, self.rect.height, path.join(IMAGE_DIR, f"{id}.png"), url)
        return image_init_data

    def get_info(self) -> dict:
        info = {"id": f"{self._id}P_bullet",
                "x": self.rect.x,
                "y": self.rect.y,
                "speed": self.speed,
                "rot": self.rot
                }
        return info
