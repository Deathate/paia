import numpy as np
import subprocess
import pickle
from pathlib import Path
import collections
from collections import namedtuple

NT = namedtuple(
    'STATE', 'frame, status, x, y, angle, R_sensor, L_sensor, F_sensor, B_sensor, L_T_sensor, R_T_sensor, crash_count, end_x, end_y, check_points')


class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.control_list = {"left_PWM": 0, "right_PWM": 0}
        self.Q = collections.defaultdict(lambda: [0, 0, 0, 0])
        self.data = []
        self.states_data = []
        self.action_data = []
        self.reward_data = []
        self.action = 0
        self.keep = 0
        self.epsilon = 0
        if (Path(__file__).parents[1]/"data/t1").exists():
            with open(Path(__file__).parents[1]/"data/t1", "rb") as f:
                for key, value in pickle.load(f).items():
                    self.Q[key] = value
        # self.max_action = 0
        # if (Path(__file__).parents[1]/"data/r1").exists():
        #     with open(Path(__file__).parents[1]/"data/r1", "rb") as f:
        #         self.max_action = pickle.load(f)

    def sarsa_update_value(self, state, action, reward, next_state, next_action, Q, done):
        gamma = 1.0
        alpha = 0.3
        # print(state, action, Q[state][action])
        Q[state][action] += alpha * \
            (reward + gamma * max(Q[next_state])
             * (1-done) - Q[state][action])

        # print(Q[state][action])

    def get_state(self, d):
        return (int(self.data[d].x),int(self.data[d].y),int(self.data[d].angle)/10)

    def get_action(self, d):
        return self.action_data[d]

    def update(self, scene_info: dict, keyboard: list = [], *args, **kwargs):
        if self.keep > 0:
            self.keep -= 1
            return self.control_list

        if len(self.data) >= 2:
            # check = -(len(self.data[-1].check_points) -
            #           len(self.data[-2].check_points))
            crash = self.data[-1].crash_count - self.data[-2].crash_count
            # reward = check - crash

            reward = -1/(min([self.data[-1].F_sensor, self.data[-1].R_sensor, self.data[-1].L_sensor,
                         self.data[-1].F_sensor, self.data[-1].B_sensor, self.data[-1].L_T_sensor, self.data[-1].R_T_sensor])+1e-1)
            # print(reward)
            self.reward_data.append(reward)
            self.sarsa_update_value(self.get_state(-2), self.get_action(-2),
                                    reward, self.get_state(-1), self.get_action(-1), self.Q, False)
            # states = self.states_data
            # actions = self.action_data
            # rewards = self.reward_data
            # dones = [False]*(len(rewards)-1) + [True]
            # print(states)
            # print(actions)
            # print(rewards)
            # for i in range(len(states)-1, 0, -1):
            #     self.sarsa_update_value(states[i-1], actions[i-1],
            #                             rewards[i-1], states[i], actions[i], self.Q, dones[i-1])
            # print(rewards)
            # self.states_data = []
            # self.action_data = []
            # self.reward_data = []

            # with open(Path(__file__).parents[1]/"data/r1", "wb") as f:
            #     pickle.dump(max(self.max_action, len(self.action_data)), f)
            # exit()

        self.data.append(NT(**scene_info))
        # self.states_data.append(self.get_state(-1))
        if np.random.rand() < 0:
            action = np.random.randint(4)
        else:
            action = np.argmax(self.Q[self.get_state(-1)])
            # print(self.Q[self.get_state(-1)])

            # print(self.get_state(-1))
            # s = int(np.sum(self.Q[self.get_state(-1)]))
            # if s!=0:
            #     print(self.Q[self.get_state(-1)])
        self.action_data.append(action)
        # print(self.get_state(-1), action, self.Q[self.get_state(-1)])
        s = 200
        if action == 0:
            self.control_list["left_PWM"] = s
            self.control_list["right_PWM"] = s
        elif action == 1:
            self.control_list["right_PWM"] = s/10
            self.control_list["left_PWM"] = -s/10
        elif action == 2:
            self.control_list["right_PWM"] = -s/10
            self.control_list["left_PWM"] = s/10
        elif action == 3:
            self.control_list["left_PWM"] = -s/2
            self.control_list["right_PWM"] = -s/2
        # self.keep = 9
        return self.control_list

    def reset(self):
        with open(Path(__file__).parents[1]/"data/t1", "wb") as f:
            pickle.dump(dict(self.Q), f)


if __name__ == "__main__":
    subprocess.Popen(
        f"python -m mlgame -f 630 -i dont_touch-master/ml/ml_play_rl.py dont_touch-master --time_to_play 500 --map 5 --sound off")
    # p = None
    # while True and not keyboard.is_pressed("esc"):
    #     if p is None:
    #         p = subprocess.Popen(
    #             f"python -m mlgame -f 660 -i dont_touch-master/ml/ml_play_rl.py dont_touch-master --time_to_play 4800 --map 5 --sound off")
    #     if p.poll() is not None:
    #         p = None
