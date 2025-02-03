import torch
import torch.optim as optim
import torch.nn.functional as F
from model import DinoDQN
from trex import TRexRunner
import random
import numpy as np
import os
import logging

# 配置日志
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# 超参数
GAMMA = 0.99
EPSILON_START = 1
EPSILON_END = 0.1
EPSILON_DECAY = 100000
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
TARGET_UPDATE = 1000
RECORD_INTERVAL = 10
EPISODE = 1000
SAVE_INTERVAL = 100  # 每隔100个episode保存一次模型

# 确保文件夹 ./models 和 ./record 存在
os.makedirs("./models", exist_ok=True)
os.makedirs("./record", exist_ok=True)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    env = TRexRunner()
    input_shape = (1, 4, 350, 1000)
    num_actions = 3

    policy_net = DinoDQN(input_shape, num_actions).to(device)
    target_net = DinoDQN(input_shape, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(10000)

    steps_done = 0
    epsilon = EPSILON_START

    for episode in range(EPISODE):
        state = env.begin()
        total_reward = 0
        record = episode % RECORD_INTERVAL == 0

        while True:
            steps_done += 1
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * steps_done / EPSILON_DECAY)
            if random.random() > epsilon:
                with torch.no_grad():
                    action = policy_net(torch.tensor(np.expand_dims(state, 0), dtype=torch.float32).to(device)).max(1)[1].view(1, 1).item()
            else:
                action = random.randrange(num_actions)

            next_state, reward, done = env.step(action, record=record, record_path=f"./record/{episode}.gif")
            total_reward += reward

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if done or env.over:
                break

            if len(replay_buffer) > BATCH_SIZE:
                transitions = replay_buffer.sample(BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32).to(device)
                batch_action = torch.tensor(np.array(batch_action), dtype=torch.int64).unsqueeze(1).to(device)
                batch_reward = torch.tensor(np.array(batch_reward), dtype=torch.float32).to(device)
                batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(device)
                batch_done = torch.tensor(np.array(batch_done), dtype=torch.float32).to(device)

                current_q_values = policy_net(batch_state).gather(1, batch_action)
                next_q_values = target_net(batch_next_state).max(1)[0].detach()
                expected_q_values = batch_reward + (GAMMA * next_q_values * (1 - batch_done))

                loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        logging.info(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}, Steps: {steps_done}")

        print(f"Episode {episode}, Total Reward: {total_reward}")
        
        if episode % SAVE_INTERVAL == 0:
            policy_net.save(f"./models/policy_net_{episode}.pth")
            target_net.save(f"./models/target_net_{episode}.pth")

        if env.over:
            break

    env.close()

if __name__ == "__main__":
    train()
