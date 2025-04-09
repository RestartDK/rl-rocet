import random
import numpy as np
import torch
import utils
import torch.optim as optim

import torch.nn as nn

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


class PositionalMapping(nn.Module):
    def __init__(self, input_dim, L=5, scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        self.scale = scale

    def forward(self, x):
        x = x * self.scale
        if self.L == 0:
            return x
        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x)
            x_cos = torch.cos(2**i * PI * x)
            h.append(x_sin)
            h.append(x_cos)
        return torch.cat(h, dim=-1) / self.scale


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_size=128, L=7):
        super().__init__()
        self.mapping = PositionalMapping(input_dim=input_dim, L=L)
        k = 1; self.add_module('linear'+str(k), nn.Linear(self.mapping.output_dim, hidden_size))
        for _ in range(hidden_layers):
            k += 1; self.add_module('linear'+str(k), nn.Linear(hidden_size, hidden_size))
        k += 1; self.add_module('linear'+str(k), nn.Linear(hidden_size, output_dim))
        self.layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = x.view([1, -1])
        x = self.mapping(x)
        for k in range(len(self.layers)-1):
            x = self.relu(self.layers[k](x))
        x = self.layers[-1](x)
        return x


GAMMA = 0.99


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_size=128, L=7, learning_rate=5e-5):
        super().__init__()
        self.output_dim = output_dim
        self.actor = MLP(input_dim, output_dim, hidden_layers, hidden_size, L)
        self.critic = MLP(input_dim, 1, hidden_layers, hidden_size, L)
        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)

    def forward(self, x):
        y = self.actor(x)
        probs = self.softmax(y)
        value = self.critic(x)
        return probs, value

    def get_action(self, state, deterministic=False, exploration=0.01):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs, value = self.forward(state)
        probs = probs[0, :]
        value = value[0]
        if deterministic:
            action_id = np.argmax(probs.detach().cpu().numpy())
        else:
            if random.random() < exploration:
                action_id = random.randint(0, self.output_dim - 1)
            else:
                action_id = np.random.choice(self.output_dim, p=np.squeeze(probs.detach().cpu().numpy()))
        log_prob = torch.log(probs[action_id] + 1e-9)
        return action_id, log_prob, value

    @staticmethod
    def update_ac(network, rewards, log_probs, values, masks, Qval, gamma=GAMMA, entropy_coeff=0.001):
        Qvals = calculate_returns(Qval.detach(), rewards, masks, gamma)
        Qvals = torch.tensor(Qvals, dtype=torch.float32).to(device).detach()

        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        # Entropy regularization
        entropy_term = -log_probs.mean()
        ac_loss = actor_loss + critic_loss - entropy_coeff * entropy_term

        network.optimizer.zero_grad()
        ac_loss.backward()
        network.optimizer.step()
