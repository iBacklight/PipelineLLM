# Q-Learning Algorithm Implementation

This directory contains a comprehensive implementation of the Q-Learning algorithm, a model-free reinforcement learning method that learns optimal actions through trial and error.

## Overview

Q-Learning is a value-based reinforcement learning algorithm that learns the quality of actions in a given state. It uses the following update rule:

```
Q(s,a) = Q(s,a) + α[r + γ * max_a'(Q(s',a')) - Q(s,a)]
```

Where:
- **α (alpha)**: Learning rate
- **γ (gamma)**: Discount factor  
- **r**: Immediate reward
- **s**: Current state
- **a**: Action taken
- **s'**: Next state

## Files

- `q_learning.py`: Core Q-learning implementation with GridWorld environment
- `train_qlearning.py`: Training script with experiments and visualizations
- `README.md`: This documentation file

## Features

### QLearningAgent Class
- **Epsilon-greedy exploration**: Balances exploration vs exploitation
- **Configurable hyperparameters**: Learning rate, discount factor, epsilon decay
- **Q-table management**: Save/load trained Q-tables
- **Training statistics**: Track rewards, episode lengths, and Q-value changes

### GridWorld Environment
- **Customizable grid size**: Any N×N grid
- **Obstacle support**: Define impassable cells
- **Goal-oriented**: Agent must reach the goal position
- **Visualization**: Render environment and learned policies

### Training Features
- **Multiple environments**: Simple, complex, and open field mazes
- **Hyperparameter comparison**: Test different learning settings
- **Progress visualization**: Plot training curves and learned policies
- **Performance testing**: Evaluate trained agents
- **Result saving**: Save Q-tables and training statistics

## Quick Start

### Basic Usage

```python
from q_learning import QLearningAgent, GridWorld

# Create environment
env = GridWorld(size=5, obstacles=[(1,1), (2,2)])

# Create agent
agent = QLearningAgent(
    state_size=env.state_size,
    action_size=env.action_size,
    learning_rate=0.1,
    discount_factor=0.95
)

# Train agent
from q_learning import train_q_learning
stats = train_q_learning(env, agent, num_episodes=1000)

# Test learned policy
state = env.reset()
for step in range(50):
    action = agent.choose_action(state, training=False)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    if done:
        print("Goal reached!")
        break
```

### Using the Training Script

```bash
# Train on simple maze
python train_qlearning.py --env simple --episodes 1000

# Train on complex maze
python train_qlearning.py --env complex --episodes 2000

# Train on all environments
python train_qlearning.py --env all --episodes 1000

# Compare hyperparameters
python train_qlearning.py --compare

# Train without saving results
python train_qlearning.py --env simple --episodes 500 --no-save
```

## Examples

### Example 1: Simple Maze

```python
import numpy as np
from q_learning import QLearningAgent, GridWorld, train_q_learning, visualize_policy

# Create a 5x5 maze with obstacles
obstacles = [(1,1), (1,2), (2,2), (3,1), (3,3)]
env = GridWorld(size=5, obstacles=obstacles)

# Create and train agent
agent = QLearningAgent(
    state_size=env.state_size,
    action_size=env.action_size,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01
)

# Train for 1000 episodes
stats = train_q_learning(env, agent, num_episodes=1000)

# Visualize learned policy
visualize_policy(env, agent)
```

### Example 2: Custom Environment

```python
# Create custom environment
env = GridWorld(size=4, obstacles=[])

# Create agent with custom parameters
agent = QLearningAgent(
    state_size=env.state_size,
    action_size=env.action_size,
    learning_rate=0.2,      # Higher learning rate
    discount_factor=0.9,    # Lower discount factor
    epsilon=0.5,            # Start with less exploration
    epsilon_decay=0.99,     # Slower decay
    epsilon_min=0.05        # Higher minimum exploration
)

# Train and test
stats = train_q_learning(env, agent, num_episodes=500)
```

### Example 3: Hyperparameter Tuning

```python
# Test different learning rates
learning_rates = [0.05, 0.1, 0.2, 0.3]
results = {}

for lr in learning_rates:
    agent = QLearningAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        learning_rate=lr,
        discount_factor=0.95
    )
    
    stats = train_q_learning(env, agent, num_episodes=500, verbose=False)
    final_rewards = stats['episode_rewards'][-100:]
    results[lr] = np.mean(final_rewards)

print("Learning Rate Comparison:")
for lr, avg_reward in results.items():
    print(f"  LR={lr}: Avg Reward={avg_reward:.2f}")
```

## Hyperparameters

### QLearningAgent Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.1 | How quickly the agent learns (0-1) |
| `discount_factor` | 0.95 | Importance of future rewards (0-1) |
| `epsilon` | 1.0 | Initial exploration rate (0-1) |
| `epsilon_decay` | 0.995 | Rate of exploration decay (0-1) |
| `epsilon_min` | 0.01 | Minimum exploration rate (0-1) |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_episodes` | 1000 | Number of training episodes |
| `max_steps_per_episode` | 100 | Maximum steps per episode |
| `verbose` | True | Print training progress |

## Environment Details

### GridWorld Actions
- **0**: Up (-1, 0)
- **1**: Down (1, 0)  
- **2**: Left (0, -1)
- **3**: Right (0, 1)

### Rewards
- **Goal reached**: +10.0
- **Hit wall**: -0.1
- **Hit obstacle**: -0.5
- **Normal move**: -0.01 (encourages efficiency)

### State Representation
States are represented as (row, column) tuples and converted to indices for Q-table lookup.

## Visualization

The implementation includes several visualization features:

1. **Training Progress**: Plot episode rewards, lengths, epsilon decay, and Q-value changes
2. **Learned Policy**: Visualize the optimal action for each state
3. **Environment Rendering**: Display the grid with obstacles, agent, and goal

## Performance Tips

1. **Learning Rate**: Start with 0.1, adjust based on convergence speed
2. **Discount Factor**: Use 0.9-0.99 for long-term planning
3. **Epsilon Decay**: Slower decay (0.99) for more exploration
4. **Episode Count**: More episodes for complex environments
5. **Grid Size**: Larger grids require more training episodes

## Troubleshooting

### Common Issues

1. **Agent not learning**: Check learning rate and epsilon decay
2. **Poor convergence**: Increase training episodes or adjust hyperparameters
3. **Over-exploration**: Increase epsilon decay rate
4. **Under-exploration**: Decrease epsilon decay rate

### Debugging

```python
# Check Q-table values
print("Q-table shape:", agent.q_table.shape)
print("Q-table range:", agent.q_table.min(), "to", agent.q_table.max())

# Check exploration rate
print("Current epsilon:", agent.epsilon)

# Check policy
policy = agent.get_policy()
print("Policy shape:", policy.shape)
```

## Extensions

This implementation can be extended for:

1. **Different environments**: CartPole, MountainCar, etc.
2. **Continuous states**: Function approximation with neural networks
3. **Multi-agent scenarios**: Multiple agents learning simultaneously
4. **Hierarchical RL**: Decompose complex tasks into subtasks
5. **Transfer learning**: Apply learned knowledge to new environments

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine Learning, 8(3-4), 279-292
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533

## License

This implementation is provided for educational purposes. Feel free to modify and extend for your own projects.
