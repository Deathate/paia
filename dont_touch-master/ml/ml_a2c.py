import keyboard
from collections import namedtuple
from pathlib import Path
import pickle
import subprocess
import torch
from torch import nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.append(r"C:\Users\aaron\Desktop\paia\MLGame-master")


class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            # nn.Linear(64, 32),
            # nn.Tanh(),
            nn.Linear(32, n_actions),
            nn.Softmax()
        )

    def forward(self, X):
        return self.model(X)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 32),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, X):
        return self.model(X)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float)
        probs = self.model(state)
        # print(probs.detach().numpy().round(2))
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def _zip(self):
        return zip(self.log_probs,
                   self.values,
                   self.rewards,
                   self.dones)

    def __iter__(self):
        for data in self._zip():
            return data

    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data

    def __len__(self):
        return len(self.rewards)


state_dim = 9
n_actions = 3
actor = Actor(state_dim, n_actions)
critic = Critic(state_dim)
adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
memory = Memory()
PATH_a = Path(__file__).parents[1] / "data/actor.pth"
PATH_c = Path(__file__).parents[1] / "data/critic.pth"


def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)


def load_model(model, filepath):
    if Path(filepath).exists():
        model.load_state_dict(torch.load(filepath))


NT = namedtuple(
    'STATE', 'frame, status, x, y, angle, R_sensor, L_sensor, F_sensor, B_sensor, L_T_sensor, R_T_sensor, crash_count, end_x, end_y, check_points')
MAP = 5
with open(Path(__file__).parents[1] / f"data/hist{MAP}", "rb") as f:
    hist = np.array(pickle.load(f))
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(hist[:, [1, 2]])

saved_log_probs = []
saved_rewards = []


class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.control_list = {"left_PWM": 0, "right_PWM": 0}
        self.data = []
        self.keep = 0
        self.is_crash = False
        load_model(policy_net, PATH)

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
            # reward = np.mean(neigh.kneighbors(
            #     [[self.get_state(-1)[0], self.get_state(-1)[1]]], return_distance=False))
            reward = 0
            crash = len(
                self.data) >= 2 and self.data[-1].crash_count != self.data[-2].crash_count
            check = len(
                self.data) >= 2 and len(self.data[-1].check_points) != len(self.data[-2].check_points)
            if check:
                print(1)
                reward = 1
            if crash:
                reward = -1
            saved_rewards.append(reward)
            if crash:
                discount_factor = 0.99
                returns = []
                cumulative_reward = 0
                for reward in reversed(saved_rewards):
                    cumulative_reward = reward + discount_factor * cumulative_reward
                    returns.insert(0, cumulative_reward)
                eps = np.finfo(np.float32).eps.item()
                returns = torch.tensor(returns)
                # returns = (returns - returns.mean()) / (returns.std() + eps)
                log_probs = torch.stack(saved_log_probs)
                loss = (log_probs * -returns).sum()
                print(loss)
                print(returns)
                print(log_probs)
                optimizer.zero_grad()
                loss.backward()
                # print(loss.backward)
                optimizer.step()

                # self.is_crash = True
                saved_log_probs.clear()
                saved_rewards.clear()
                save_model(policy_net, PATH)
                # exit()
            # log_prob = self.action_dist.log_prob(self.get_action(-1))
            # reward = np.mean(neigh.kneighbors(
            #     [[self.get_state(-1)[0], self.get_state(-1)[1]]], return_distance=False))
            # print(policy_net.state_dict())
            # if crash:
            #     save_model(policy_net, PATH)
            #     exit()
        self.data.append(NT(**scene_info))
        action, log_prob = policy_net.act(self.get_state(-1))
        saved_log_probs.append(log_prob)

        s = 200
        if action == 0:
            self.control_list["left_PWM"] = s
            self.control_list["right_PWM"] = s
        elif action == 1:
            self.control_list["right_PWM"] = s/15
            self.control_list["left_PWM"] = 0
        elif action == 2:
            self.control_list["right_PWM"] = 0
            self.control_list["left_PWM"] = s/15
        elif action == 3:
            self.control_list["right_PWM"] = -s/2
            self.control_list["left_PWM"] = -s/2
        self.keep = 9
        return self.control_list

    def reset(self):
        pass


if __name__ == "__main__":
    p = subprocess.Popen(
        f"python -m mlgame -f 200 -i dont_touch-master/ml/ml_torch.py dont_touch-master --time_to_play 4800 --map {MAP} --sound off").wait()
    # p = None
    # while True and not keyboard.is_pressed("esc"):
    #     if p is None:
    #         p = subprocess.Popen(
    #             f"python -m mlgame -f 500 -i dont_touch-master/ml/ml_torch.py dont_touch-master --time_to_play 4800 --map 5 --sound off")
    #     if p.poll() is not None:
    #         p = None
