from os import path
from .env import SOUND_DIR
from .GameFramework.SoundController import SoundController


class TankSoundController(SoundController):
    BGM_PATH = path.join(SOUND_DIR, "BGM.ogg")
    SHOOT_PATH = path.join(SOUND_DIR, "shoot.wav")
    TOUCH_PATH = path.join(SOUND_DIR, "touch.wav")

    def play_bgm(self):
        self.play_music(self.BGM_PATH, 0.1)

    def play_shoot_sound(self):
        self.play_sound(self.SHOOT_PATH, 0.03, -1)

    def play_touch_sound(self):
        self.play_sound(self.TOUCH_PATH, 0.1, -1)


