import keyboard
from collections import namedtuple
from pathlib import Path
import pickle
import subprocess
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.append(r"C:\Users\aaron\Desktop\paia\MLGame-master")


class PolicyNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hsize):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hsize)
        self.fc2 = torch.nn.Linear(hsize, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float)
        probs = policy_net(state)
        # print(probs.detach().numpy().round(2))
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


policy_net = PolicyNet(input_dim=9, output_dim=4, hsize=64)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
PATH = Path(__file__).parents[1] / "data/pg.pth"


def save_model(filepath):
    torch.save({"model": policy_net.state_dict(),
                "optimizer": optimizer.state_dict()}, filepath)


def load_model(filepath):
    if filepath.exists():
        checkpoint = torch.load(filepath)
        policy_net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])


NT = namedtuple(
    'STATE', 'frame, status, x, y, angle, R_sensor, L_sensor, F_sensor, B_sensor, L_T_sensor, R_T_sensor, crash_count, end_x, end_y, check_points')
MAP = 5
with open(Path(__file__).parents[1] / f"data/hist{MAP}", "rb") as f:
    hist = np.array(pickle.load(f))
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(hist[:, [1, 2]])

saved_log_probs = []
saved_rewards = []

max_len = 3


class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.control_list = {"left_PWM": 0, "right_PWM": 0}
        self.data = []
        self.keep = 0
        load_model(PATH)

    def get_state(self, d):
        return (self.data[d].x/1000, self.data[d].y/1000, self.data[d].angle/360, self.data[d].R_sensor/100,
                self.data[d].L_sensor/100, self.data[d].F_sensor /
                100, self.data[d].B_sensor/100,
                self.data[d].L_T_sensor/100, self.data[d].R_T_sensor/100)

    def update(self, scene_info: dict, keyboard: list = [], *args, **kwargs):
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        if self.keep > 0:
            self.keep -= 1
            return self.control_list

        if len(self.data) >= 1:
            reward = -1/(min([self.data[-1].F_sensor, self.data[-1].R_sensor, self.data[-1].L_sensor,
                         self.data[-1].F_sensor, self.data[-1].B_sensor, self.data[-1].L_T_sensor, self.data[-1].R_T_sensor])+1e-1)
            # reward = np.mean(neigh.kneighbors(
            #     [[self.get_state(-1)[0]*1000, self.get_state(-1)[1]*1000]], return_distance=False))
            # crash = len(
            #     self.data) >= 2 and self.data[-1].crash_count != self.data[-2].crash_count
            # check = len(
            #     self.data) >= 2 and len(self.data[-1].check_points) != len(self.data[-2].check_points)

            saved_rewards.append(reward)
            returns = torch.tensor(saved_rewards[-1], dtype=float)
            # eps = np.finfo(np.float32).eps.item()
            # returns = (returns - returns.mean()) / (returns.std() + eps)
            log_probs = saved_log_probs[-1]
            loss = -log_probs * returns
            print(reward, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            saved_log_probs.clear()
            saved_rewards.clear()
            # save_model(PATH)

        self.data.append(NT(**scene_info))
        action, log_prob = policy_net.act(self.get_state(-1))
        saved_log_probs.append(log_prob)

        s = 100
        if action == 0:
            self.control_list["left_PWM"] = s
            self.control_list["right_PWM"] = s
        elif action == 1:
            self.control_list["right_PWM"] = s/10
            self.control_list["left_PWM"] = 0
        elif action == 2:
            self.control_list["right_PWM"] = 0
            self.control_list["left_PWM"] = s/10
        elif action == 3:
            self.control_list["right_PWM"] = -s/2
            self.control_list["left_PWM"] = -s/2
        self.keep = 9
        return self.control_list

    def reset(self):
        save_model(PATH)


if __name__ == "__main__":
    p = subprocess.Popen(
        f"python -m mlgame -f 500 -i dont_touch-master/ml/ml_pg.py dont_touch-master --time_to_play 500 --map {MAP} --sound off")
    # p = None
    # while True and not keyboard.is_pressed("esc"):
    #     if p is None:
    #         p = subprocess.Popen(
    #             f"python -m mlgame -f 500 -i dont_touch-master/ml/ml_torch.py dont_touch-master --time_to_play 4800 --map 5 --sound off")
    #     if p.poll() is not None:
    #         p = None
