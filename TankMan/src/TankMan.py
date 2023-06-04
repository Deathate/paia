import sys
from os import path

from mlgame.view.view_model import Scene

from .TankBattleMode import TankBattleMode
from .env import MAP_DIR
from .GameFramework.MyGame import GameFramework

'''need some fuction same as arkanoid which without dash in the name of fuction'''


class TankMan(GameFramework):
    def __init__(self, user_num: int, station_num: int, map_no: int, frame_limit: int, sound: str):
        self.station_num = station_num
        super().__init__(user_num, map_no, frame_limit, sound)
        self.game_mode = self.set_game_mode()
        self.scene = Scene(width=self.game_mode.map_width, height=self.game_mode.map_height+100, color="#ffffff", bias_y=50)

    def set_game_mode(self):
        map_path = path.join(MAP_DIR, self.map_name)
        game_mode = TankBattleMode(self._user_num, self.station_num, map_path, self.frame_limit, self.is_sound)
        return game_mode
