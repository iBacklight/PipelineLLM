#!/usr/bin/env python3
"""
Training script for Q-Learning algorithm.

This script demonstrates how to train a Q-learning agent on different
GridWorld environments and visualize the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from q_learning import QLearningAgent, GridWorld, train_q_learning, plot_training_progress, visualize_policy
import argparse
import os


def create_simple_maze():
    """Create a simple maze environment."""
    obstacles = [
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 3),
        (3, 1), (3, 2), (3, 3)
    ]
    return GridWorld(size=5, obstacles=obstacles)


def create_complex_maze():
    """Create a more complex maze environment."""
    obstacles = [
        (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 1), (2, 3), (2, 4),
        (3, 1), (3, 2), (3, 3), (3, 4),
        (4, 1), (4, 2)
    ]
    return GridWorld(size=6, obstacles=obstacles)


def create_open_field():
    """Create an open field with no obstacles."""
    return GridWorld(size=4, obstacles=[])


def run_experiment(env_name: str, num_episodes: int = 1000, save_results: bool = True):
    """
    Run a Q-learning experiment on a specific environment.
    
    Args:
        env_name: Name of the environment ('simple', 'complex', 'open')
        num_episodes: Number of training episodes
        save_results: Whether to save results to files
    """
    print(f"\n{'='*60}")
    print(f"Q-Learning Experiment: {env_name.upper()} MAZE")
    print(f"{'='*60}")
    
    # Create environment
    if env_name == 'simple':
        env = create_simple_maze()
    elif env_name == 'complex':
        env = create_complex_maze()
    elif env_name == 'open':
        env = create_open_field()
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
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
    
    print(f"Environment: {env.size}x{env.size} grid")
    
    # Train agent
    print("Training agent...")
    stats = train_q_learning(env, agent, num_episodes=num_episodes, verbose=True)
    
    # Calculate final performance
    final_rewards = stats['episode_rewards'][-100:]
    final_lengths = stats['episode_lengths'][-100:]
    
    print(f"\nFinal Performance (last 100 episodes):")
    print(f"  Average reward: {np.mean(final_rewards):.3f} ± {np.std(final_rewards):.3f}")
    print(f"  Average length: {np.mean(final_lengths):.1f} ± {np.std(final_lengths):.1f}")
    print(f"  Success rate: {np.mean([r > 0 for r in final_rewards]):.1%}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_training_progress(stats)
    visualize_policy(env, agent)
    
    # Test the learned policy
    print("\nTesting learned policy...")
    test_agent(env, agent, num_tests=5)
    
    # Save results if requested
    if save_results:
        save_experiment_results(env_name, agent, stats)
    
    return agent, stats


def test_agent(env: GridWorld, agent: QLearningAgent, num_tests: int = 5):
    """Test the trained agent multiple times."""
    print(f"\nRunning {num_tests} test episodes...")
    
    test_results = []
    for test in range(num_tests):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100
        
        for step in range(max_steps):
            action = agent.choose_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        test_results.append({
            'reward': total_reward,
            'steps': steps,
            'success': done
        })
        
        status = "SUCCESS" if done else "FAILED"
        print(f"  Test {test + 1}: {status} - Reward: {total_reward:.2f}, Steps: {steps}")
    
    # Summary
    success_rate = np.mean([r['success'] for r in test_results])
    avg_reward = np.mean([r['reward'] for r in test_results])
    avg_steps = np.mean([r['steps'] for r in test_results])
    
    print(f"\nTest Summary:")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average steps: {avg_steps:.1f}")


def save_experiment_results(env_name: str, agent: QLearningAgent, stats: dict):
    """Save experiment results to files."""
    # Create results directory
    results_dir = f"results_{env_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save Q-table
    q_table_path = os.path.join(results_dir, "q_table.npy")
    agent.save_q_table(q_table_path)
    print(f"Q-table saved to: {q_table_path}")
    
    # Save training statistics
    stats_path = os.path.join(results_dir, "training_stats.npz")
    np.savez(stats_path, **stats)
    print(f"Training statistics saved to: {stats_path}")
    
    # Save policy visualization
    policy_path = os.path.join(results_dir, "policy.txt")
    policy = agent.get_policy()
    action_symbols = ['↑', '↓', '←', '→']
    
    with open(policy_path, 'w') as f:
        f.write(f"Learned Policy for {env_name} maze:\n")
        f.write("=" * 50 + "\n")
        f.write("Legend: ↑=Up, ↓=Down, ←=Left, →=Right, X=Obstacle, G=Goal\n\n")
        
        # This is a simplified text representation
        # For full visualization, use the visualize_policy function
        f.write("Policy matrix (action indices):\n")
        for i in range(int(np.sqrt(agent.state_size))):
            row = []
            for j in range(int(np.sqrt(agent.state_size))):
                state_idx = i * int(np.sqrt(agent.state_size)) + j
                row.append(str(policy[state_idx]))
            f.write(" ".join(row) + "\n")
    
    print(f"Policy saved to: {policy_path}")


def compare_hyperparameters():
    """Compare different hyperparameter settings."""
    print(f"\n{'='*60}")
    print("HYPERPARAMETER COMPARISON")
    print(f"{'='*60}")
    
    env = create_simple_maze()
    
    # Different hyperparameter configurations
    configs = [
        {"name": "High Learning Rate", "learning_rate": 0.3, "discount_factor": 0.9},
        {"name": "Low Learning Rate", "learning_rate": 0.05, "discount_factor": 0.9},
        {"name": "High Discount", "learning_rate": 0.1, "discount_factor": 0.99},
        {"name": "Low Discount", "learning_rate": 0.1, "discount_factor": 0.8},
        {"name": "Default", "learning_rate": 0.1, "discount_factor": 0.95},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Discount factor: {config['discount_factor']}")
        
        agent = QLearningAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            learning_rate=config['learning_rate'],
            discount_factor=config['discount_factor'],
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        # Train for fewer episodes for comparison
        stats = train_q_learning(env, agent, num_episodes=500, verbose=False)
        
        # Calculate final performance
        final_rewards = stats['episode_rewards'][-50:]
        success_rate = np.mean([r > 0 for r in final_rewards])
        avg_reward = np.mean(final_rewards)
        
        results[config['name']] = {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'final_epsilon': agent.epsilon
        }
        
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Average reward: {avg_reward:.2f}")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("HYPERPARAMETER COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Configuration':<20} {'Success Rate':<12} {'Avg Reward':<12} {'Final Epsilon':<12}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<20} {result['success_rate']:<12.1%} {result['avg_reward']:<12.2f} {result['final_epsilon']:<12.3f}")


def main():
    """Main function to run Q-learning experiments."""
    parser = argparse.ArgumentParser(description='Train Q-Learning agent on GridWorld')
    parser.add_argument('--env', choices=['simple', 'complex', 'open', 'all'], default='simple',
                       help='Environment to train on')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Number of training episodes')
    parser.add_argument('--compare', action='store_true', default=False,
                       help='Run hyperparameter comparison')
    parser.add_argument('--no-save', action='store_true', default=False,    
                       help='Do not save results to files')
    
    args = parser.parse_args()
    
    print("Q-Learning Training Script")
    print("=" * 50)
    
    if args.compare:
        compare_hyperparameters()
        return
    
    if args.env == 'all':
        environments = ['simple', 'complex', 'open']
        for env_name in environments:
            run_experiment(env_name, args.episodes, not args.no_save)
    else:
        run_experiment(args.env, args.episodes, not args.no_save)
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
