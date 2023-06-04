import subprocess
import pickle
from pathlib import Path
import pygame

hist = []


class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.control_list = {"left_PWM": 0, "right_PWM": 0}
        self.map = kwargs["game_params"]["map"]

    def update(self, scene_info: dict, keyboard: list = [], *args, **kwargs):
        self.control_list["left_PWM"] = self.control_list["right_PWM"] = 0
        frame, status, x, y, angle, R_sensor, L_sensor, F_sensor, B_sensor, L_T_sensor, R_T_sensor, crash_count, end_x, end_y, check_points = scene_info.values()

        # self.control_list["left_PWM"] = 1
        # self.control_list["right_PWM"] = 1
        s = 100
        action = -1
        if pygame.K_w in keyboard or pygame.K_UP in keyboard:
            self.control_list["left_PWM"] = s
            self.control_list["right_PWM"] = s
            action = 1
        elif pygame.K_a in keyboard or pygame.K_LEFT in keyboard:
            self.control_list["right_PWM"] = s/10
            action = 2
        elif pygame.K_d in keyboard or pygame.K_RIGHT in keyboard:
            self.control_list["left_PWM"] = s/10
            action = 3
        elif pygame.K_s in keyboard or pygame.K_DOWN in keyboard:
            self.control_list["left_PWM"] = -s
            self.control_list["right_PWM"] = -s
            action = 4
        elif pygame.K_q in keyboard:
            self.reset()
            exit()
        # if action != -1:
        hist.append((frame, x, y, angle, R_sensor, L_sensor,
                    F_sensor, B_sensor, L_T_sensor, R_T_sensor, end_x, end_y, action))

        return self.control_list

    def reset(self):
        with open(Path(__file__).parents[0] / f"hist{self.map}.pickle", "wb") as f:
            pickle.dump(hist, f)


if __name__ == "__main__":
    subprocess.Popen(
        f"python -m mlgame -f 60 -i dont_touch-master/ml/ml_play_collect.py dont_touch-master --time_to_play 4800 --map 10 --sound off").wait()
