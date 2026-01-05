"""
Q-Learning Algorithm Implementation

This module implements the Q-learning algorithm, a model-free reinforcement learning
algorithm that learns the value of actions in a given state.

Q-Learning Update Rule:
Q(s,a) = Q(s,a) + α[r + γ * max_a'(Q(s',a')) - Q(s,a)]

Where:
- α (alpha) is the learning rate
- γ (gamma) is the discount factor
- r is the immediate reward
- s is the current state
- a is the action taken
- s' is the next state
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import random
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning Agent implementation.
    
    This agent learns an optimal policy by iteratively updating Q-values
    using the Q-learning update rule.
    """
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning Agent.
        
        Args:
            state_size: Number of possible states
            action_size: Number of possible actions
            learning_rate: Learning rate (α)
            discount_factor: Discount factor (γ)
            epsilon: Initial exploration rate
            epsilon_decay: Rate of epsilon decay
            epsilon_min: Minimum epsilon value
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))
        
        # Track training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.q_value_changes = []
    
    def get_state_index(self, state) -> int:
        """
        Convert state to index for Q-table lookup.
        Override this method for custom state representations.
        """
        if isinstance(state, (int, np.integer)):
            return int(state)
        elif isinstance(state, tuple):
            # For 2D states like (row, col)
            return state[0] * int(np.sqrt(self.state_size)) + state[1]
        else:
            return hash(state) % self.state_size
    
    def choose_action(self, state, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether we're in training mode (affects exploration)
            
        Returns:
            Action index
        """
        state_idx = self.get_state_index(state)
        
        if training and random.random() < self.epsilon:
            # Explore: choose random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploit: choose best action
            return np.argmax(self.q_table[state_idx])
    
    def update_q_value(self, state, action: int, reward: float, next_state, done: bool):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        state_idx = self.get_state_index(state)
        next_state_idx = self.get_state_index(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_idx, action]
        
        if done:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Non-terminal state: add discounted future reward
            max_next_q = np.max(self.q_table[next_state_idx])
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-learning update
        q_change = self.learning_rate * (target_q - current_q)
        self.q_table[state_idx, action] += q_change
        
        # Track Q-value changes for analysis
        self.q_value_changes.append(abs(q_change))
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_policy(self) -> np.ndarray:
        """
        Get the current policy (greedy action for each state).
        
        Returns:
            Array of best actions for each state
        """
        return np.argmax(self.q_table, axis=1)
    
    def get_q_values(self, state) -> np.ndarray:
        """
        Get Q-values for a given state.
        
        Args:
            state: State to get Q-values for
            
        Returns:
            Array of Q-values for all actions in the state
        """
        state_idx = self.get_state_index(state)
        return self.q_table[state_idx]
    
    def save_q_table(self, filepath: str):
        """Save Q-table to file."""
        np.save(filepath, self.q_table)
    
    def load_q_table(self, filepath: str):
        """Load Q-table from file."""
        self.q_table = np.load(filepath)


class GridWorld:
    """
    Simple GridWorld environment for testing Q-learning.
    
    The agent starts at (0,0) and must reach the goal at (n-1, n-1).
    There may be obstacles and different reward structures.
    """
    
    def __init__(self, size: int = 5, obstacles: List[Tuple[int, int]] = None):
        """
        Initialize GridWorld.
        
        Args:
            size: Grid size (size x size)
            obstacles: List of obstacle positions
        """
        self.size = size
        self.state_size = size * size
        self.action_size = 4  # Up, Down, Left, Right
        
        # Define actions
        self.actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        
        # Set up obstacles
        self.obstacles = set(obstacles) if obstacles else set()
        
        # Start and goal positions
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        
        # Current position
        self.current_pos = self.start_pos
        
    def reset(self) -> Tuple[int, int]:
        """Reset environment to initial state."""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            (next_state, reward, done, info)
        """
        # Get action direction
        direction = self.actions[action]
        new_pos = (self.current_pos[0] + direction[0], 
                  self.current_pos[1] + direction[1])
        
        # Check bounds
        if (new_pos[0] < 0 or new_pos[0] >= self.size or 
            new_pos[1] < 0 or new_pos[1] >= self.size):
            # Hit wall - stay in place
            reward = -0.1
            done = False
        elif new_pos in self.obstacles:
            # Hit obstacle - stay in place
            reward = -0.5
            done = False
        elif new_pos == self.goal_pos:
            # Reached goal
            self.current_pos = new_pos
            reward = 10.0
            done = True
        else:
            # Normal move
            self.current_pos = new_pos
            reward = -0.01  # Small negative reward to encourage efficiency
            done = False
        
        return self.current_pos, reward, done, {}
    
    def render(self, q_table: np.ndarray = None, policy: np.ndarray = None):
        """
        Render the grid world.
        
        Args:
            q_table: Q-table to visualize
            policy: Policy to visualize
        """
        grid = np.zeros((self.size, self.size))
        
        # Mark obstacles
        for obs in self.obstacles:
            grid[obs] = -1
        
        # Mark goal
        grid[self.goal_pos] = 2
        
        # Mark current position
        grid[self.current_pos] = 1
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot grid
        im = ax.imshow(grid, cmap='RdYlBu', vmin=-1, vmax=2)
        
        # Add text annotations
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) in self.obstacles:
                    ax.text(j, i, 'X', ha='center', va='center', fontsize=20, color='white')
                elif (i, j) == self.goal_pos:
                    ax.text(j, i, 'G', ha='center', va='center', fontsize=20, color='white')
                elif (i, j) == self.current_pos:
                    ax.text(j, i, 'A', ha='center', va='center', fontsize=20, color='white')
                else:
                    ax.text(j, i, '.', ha='center', va='center', fontsize=16, color='black')
        
        ax.set_title('GridWorld Environment')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()
    
    def get_state_index(self, pos: Tuple[int, int]) -> int:
        """Convert position to state index."""
        return pos[0] * self.size + pos[1]


def train_q_learning(env: GridWorld, 
                    agent: QLearningAgent, 
                    num_episodes: int = 1000,
                    max_steps_per_episode: int = 100,
                    verbose: bool = True) -> Dict:
    """
    Train Q-learning agent on the environment.
    
    Args:
        env: Environment to train on
        agent: Q-learning agent
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        verbose: Whether to print progress
        
    Returns:
        Training statistics
    """
    stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'epsilon_values': [],
        'q_value_changes': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps_per_episode):
            # Choose action
            action = agent.choose_action(state, training=True)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Update Q-value
            agent.update_q_value(state, action, reward, next_state, done)
            
            # Update statistics
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record statistics
        stats['episode_rewards'].append(total_reward)
        stats['episode_lengths'].append(steps)
        stats['epsilon_values'].append(agent.epsilon)
        
        if len(agent.q_value_changes) > 0:
            stats['q_value_changes'].append(np.mean(agent.q_value_changes[-100:]))
            agent.q_value_changes = []  # Reset for next episode
        
        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(stats['episode_rewards'][-100:])
            avg_length = np.mean(stats['episode_lengths'][-100:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Length: {avg_length:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return stats


def plot_training_progress(stats: Dict):
    """Plot training progress."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(stats['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(stats['episode_lengths'])
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Epsilon decay
    axes[1, 0].plot(stats['epsilon_values'])
    axes[1, 0].set_title('Epsilon Decay')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].grid(True)
    
    # Q-value changes
    if stats['q_value_changes']:
        axes[1, 1].plot(stats['q_value_changes'])
        axes[1, 1].set_title('Q-Value Changes')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Mean |Q-Change|')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


def visualize_policy(env: GridWorld, agent: QLearningAgent):
    """Visualize the learned policy."""
    policy = agent.get_policy()
    
    # Create policy grid
    policy_grid = np.zeros((env.size, env.size))
    action_symbols = ['↑', '↓', '←', '→']
    
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) in env.obstacles:
                policy_grid[i, j] = -1
            elif (i, j) == env.goal_pos:
                policy_grid[i, j] = 2
            else:
                state_idx = env.get_state_index((i, j))
                policy_grid[i, j] = policy[state_idx]
    
    # Plot policy
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(policy_grid, cmap='RdYlBu', vmin=-1, vmax=3)
    
    # Add action symbols
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) in env.obstacles:
                ax.text(j, i, 'X', ha='center', va='center', fontsize=20, color='white')
            elif (i, j) == env.goal_pos:
                ax.text(j, i, 'G', ha='center', va='center', fontsize=20, color='white')
            else:
                action_idx = int(policy_grid[i, j])
                ax.text(j, i, action_symbols[action_idx], ha='center', va='center', 
                       fontsize=16, color='black')
    
    ax.set_title('Learned Policy')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Q-Learning Algorithm Demo")
    print("=" * 50)
    
    # Create environment
    obstacles = [(1, 1), (2, 2), (3, 1)]  # Some obstacles
    env = GridWorld(size=5, obstacles=obstacles)
    
    # Create agent
    agent = QLearningAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print(f"Environment: {env.size}x{env.size} grid with {len(obstacles)} obstacles")
    print(f"Agent: Learning rate={agent.learning_rate}, Discount factor={agent.discount_factor}")
    print()
    
    # Train agent
    print("Training agent...")
    stats = train_q_learning(env, agent, num_episodes=1000, verbose=True)
    
    # Plot training progress
    print("\nPlotting training progress...")
    plot_training_progress(stats)
    
    # Visualize learned policy
    print("Visualizing learned policy...")
    visualize_policy(env, agent)
    
    # Test the learned policy
    print("\nTesting learned policy...")
    state = env.reset()
    total_reward = 0
    steps = 0
    max_steps = 50
    
    print(f"Starting at {state}, goal at {env.goal_pos}")
    
    for step in range(max_steps):
        action = agent.choose_action(state, training=False)  # No exploration
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        action_names = ['Up', 'Down', 'Left', 'Right']
        print(f"Step {step + 1}: {state} -> {action_names[action]} -> {next_state} (reward: {reward:.2f})")
        
        state = next_state
        
        if done:
            print(f"Goal reached! Total reward: {total_reward:.2f}, Steps: {steps}")
            break
    else:
        print(f"Goal not reached in {max_steps} steps. Total reward: {total_reward:.2f}")
    
    print("\nQ-Learning demo completed!")
