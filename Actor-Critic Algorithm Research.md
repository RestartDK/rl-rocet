# Actor-Critic Algorithm Research

## Overview
The `policy.py` file defines a Reinforcement Learning (RL) agent using the Actor-Critic method. Actor-Critic is a hybrid RL algorithm that combines both value-based and policy-based approaches by maintaining two separate models:
* The actor, which learns the policy function (what action to take)
* The critic, which learns the value function (how good the action was)

## Core Components in the Code

### ActorCritic Class
This is the main class implementing the Actor-Critic model. It consists of:
* `self.actor` – a neural network outputting action probabilities
* `self.critic` – a neural network estimating the state-value
* Both networks use a custom MLP with optional positional encoding, a unique feature discussed below

### Forward Pass: `def forward(self, x)`
This method:
* Passes the input x through the actor to get action probabilities (via softmax)
* Passes the same input through the critic to get the state value

### Action Selection: `def get_action(...)`
This method:
* Computes action probabilities from the actor
* Selects an action either deterministically (highest probability) or stochastically (sampled)
* Adds exploration by random choice
* Logs the probability and critic-estimated value of the action

### Training: `def update_ac(...)`
This method:
* Computes the returns (Q-values) from rewards and terminal masks
* Calculates advantage: how much better/worse the action was than expected
* Computes:
  * Actor loss: Encourages actions with high advantage
  * Critic loss: Reduces prediction error in state values
* Optimizes both via gradient descent

## Unique Modality: Positional Encoding in Actor-Critic
One notable innovation in this implementation is the use of Positional Mapping via the `PositionalMapping` class.

### Why it matters:
* Inspired by NeRF (Neural Radiance Fields), this maps low-dimensional state input into a higher-dimensional Fourier feature space using sinusoids
* The purpose is to help the MLPs (both actor and critic) learn high-frequency patterns more efficiently, potentially improving generalization and policy sharpness

### Code Example:
```python
self.mapping = PositionalMapping(input_dim=input_dim, L=7)
```
This layer transforms the state input before it's passed to the MLP layers. When `L=0`, the mapping is skipped (acts like a regular MLP). With `L > 0`, it allows the network to handle more complex functions.

## Additional Implementation Insights
* Softmax over actor outputs ensures a valid probability distribution over actions
* Entropy regularization is mentioned in comments as a potential future enhancement (used in Soft Actor-Critic)
* Uses RMSProp as the default optimizer, with the option to switch to Adam
* Efficient logging of log probabilities is crucial for computing actor gradients

