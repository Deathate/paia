import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import gymnasium as gym

# helper function to convert numpy arrays to tensors
def t(x): return torch.from_numpy(x)
# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=0)
        )
    
    def act(self,state):
        probs = self.model(t(state))
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, X):
        return self.model(X)
# Memory
# Stores results from the networks, instead of calculating the operations again from states, etc.
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
def plot_res(values, title=''):
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red', ls='--', label='goal')
    ax[0].axhline(np.mean(values[-100:]), c='blue', ls='--')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
    except:
        pass
    ax[1].hist(values[-50:])
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    # ax[1].legend()
    plt.show()

env = gym.make("CartPole-v1")
env.reset(seed=1)
torch.manual_seed(1)
# config
state_dim = 4
n_actions = 2
actor = Actor(state_dim, n_actions)
critic = Critic(state_dim)
adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
gamma = 0.99
memory = Memory()
max_steps = 200
# train function
def train(memory):
    values = torch.stack(memory.values)
    q_vals = np.zeros((len(memory), 1))
    
    # target values are calculated backward
    # it's super important to handle correctly done states,
    # for those cases we want our to target to be equal to the reward only
    q_val=0
    for i, (_, _, reward, done) in enumerate(memory.reversed()):
        q_val = reward + gamma*q_val*(1.0-done)
        q_vals[len(memory)-1 - i] = q_val # store values from the end to the beginning

    advantage = torch.Tensor(q_vals) - values
    
    critic_loss = advantage.pow(2).mean()
    adam_critic.zero_grad()
    critic_loss.backward()
    adam_critic.step()
    
    actor_loss = (-torch.stack(memory.log_probs)*advantage.detach()).mean()
    adam_actor.zero_grad()
    actor_loss.backward()
    adam_actor.step()

episode_rewards = []

while True:
    done = truncated=False
    total_reward = 0
    state,_ = env.reset()

    finish = lambda: done or truncated
    while True:
        action,log_prob = actor.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        if finish():
            memory.add(log_prob, critic(t(state)), -1, done)
            train(memory)
            memory.clear()
            break
        
        total_reward += reward
        memory.add(log_prob, critic(t(state)), reward, done)
        
        state = next_state
    episode_rewards.append(total_reward)

    if np.mean(episode_rewards[-20:])>=200:
        print(len(episode_rewards))
        plot_res(episode_rewards, title='Episodic A2C')
        break
