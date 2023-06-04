import torch
import gym

# Create the environment
env = gym.make('CartPole-v1')

# Define the DQN agent


class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.dqn = torch.nn.Sequential(
            torch.nn.Linear(state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_size)
        )
        self.optimizer = torch.optim.Adam(self.dqn.parameters())
        self.loss_fn = torch.nn.MSELoss()

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action = self.dqn(state).argmax()
        return action

    def learn(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        # action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        # Predict the Q value for the current state and action
        q_value = self.dqn(state)[action]

        # Get the Q value for the next state
        next_q_value = self.dqn(next_state).max()

        # Calculate the target Q value
        target_q_value = reward + (1 - done) * next_q_value

        # Calculate the loss
        loss = self.loss_fn(q_value, target_q_value)

        # Backpropagate the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Train the agent
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
for episode in range(5000):
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action.item())
        agent.learn(state, action, reward, next_state, done)
        state = next_state
env.close()
# Test the agent
env = gym.make('CartPole-v1', render_mode='human')
for episode in range(10):
    state, _ = env.reset()
    done = False
    r = 0
    while not done:
        next_state, reward, done, _, _ = env.step(action.item())
        state = next_state
        r += reward
    print(r)

env.close()
# import gym

# env = gym.make("MountainCar-v0", render_mode='human')
# state = env.reset()

# done = False
# while not done:
#     env.step(2)
# env.close()
