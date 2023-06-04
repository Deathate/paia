import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import subprocess
import pickle
from pathlib import Path


class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.control_list = {"left_PWM": 0, "right_PWM": 0}
        map = kwargs["game_params"]["map"]
        hist = []
        for i in range(12):
            with open(Path(__file__).parents[0] / f"hist{i+1}.pickle", "rb") as f:
                r = pickle.load(f)[0]
                hist.append([r[-3], r[-2]])
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.knn.fit(hist, list(range(1, 13, 1)))
        self.a = None
    # self.a = hist[:, -1].astype(int)

    def update(self, scene_info: dict, keyboard: list = [], *args, **kwargs):
        self.control_list["left_PWM"] = self.control_list["right_PWM"] = 0
        frame, status, x, y, angle, R_sensor, L_sensor, F_sensor, B_sensor, L_T_sensor, R_T_sensor, crash_count, end_x, end_y, check_points = scene_info.values()
        if self.a is None:
            map = self.knn.predict(
                [[end_x, end_y]])[0]
            with open(Path(__file__).parents[0] / f"hist{map}.pickle", "rb") as f:
                self.a = np.array(pickle.load(f))[:, -1]
                # print(self.a)
                # exit()

        # action = self.knn.predict(
        #     [[frame, x, y, angle, R_sensor, L_sensor, F_sensor, B_sensor, L_T_sensor, R_T_sensor, end_x, end_y]])[0]
        # frame = frame % 60
        action = self.a[frame]
        # self.control_list["left_PWM"] = 1
        # self.control_list["right_PWM"] = 1
        s = 100
        if action == 1:
            self.control_list["left_PWM"] = s
            self.control_list["right_PWM"] = s
        elif action == 2:
            self.control_list["right_PWM"] = s/10
        elif action == 3:
            self.control_list["left_PWM"] = s/10
        elif action == 4:
            self.control_list["left_PWM"] = -s
            self.control_list["right_PWM"] = -s

        return self.control_list

    def reset(self):
        pass


if __name__ == "__main__":
    subprocess.Popen(
        f"python -m mlgame -f 60 -i dont_touch-master/ml/ml_play.py dont_touch-master --time_to_play 4800 --map 1 --sound off").wait()
