#!/usr/bin/env python3
"""Demo script to showcase the training loop functionality."""

import jax
import sys
sys.path.append('src')

from src.agent.ppo import PPOAgent
from src.environment.cartpole import CartPoleWrapper
from src.training.trainer import PPOTrainer


def main():
    """Main demo function."""
    print("🚀 Starting CartPole PPO Training Demo")
    print("=" * 50)
    
    # Create random key
    key = jax.random.PRNGKey(42)
    
    # Create environment
    print("📦 Creating CartPole environment...")
    env = CartPoleWrapper(max_episode_steps=200)
    
    # Create agent
    print("🤖 Initializing PPO agent...")
    agent = PPOAgent(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        learning_rate=3e-4,
        batch_size=8,
        minibatch_size=4,
        epochs_per_update=4,
        key=key
    )
    
    # Create trainer
    print("🏃‍♂️ Setting up training loop...")
    trainer = PPOTrainer(
        agent=agent,
        env=env,
        max_episodes=20,
        max_steps_per_episode=200,
        target_reward=150.0,  # Lower target for demo
        convergence_window=5,
        early_stopping_patience=10,
        eval_frequency=5,
        eval_episodes=3,
        log_frequency=1,
        key=key
    )
    
    # Training callback
    def training_callback(metrics):
        episode = metrics['episode']
        step_metrics = metrics['step_metrics']
        
        if episode % 5 == 0:
            print(f"  📊 Episode {episode}: Reward = {step_metrics['avg_reward']:.1f}, "
                  f"Length = {step_metrics['avg_length']:.1f}")
        
        if metrics['eval_metrics']:
            eval_metrics = metrics['eval_metrics']
            print(f"  🎯 Evaluation: Reward = {eval_metrics['avg_reward']:.1f} ± "
                  f"{eval_metrics['std_reward']:.1f}")
    
    # Run training
    print("\n🎯 Starting training...")
    results = trainer.train(callback=training_callback)
    
    # Display results
    print("\n" + "=" * 50)
    print("📈 Training Results:")
    print(f"  Total episodes: {results['total_episodes']}")
    print(f"  Total steps: {results['total_steps']}")
    print(f"  Training time: {results['training_time']:.2f} seconds")
    print(f"  Final evaluation reward: {results['final_avg_reward']:.2f} ± {results['final_std_reward']:.2f}")
    print(f"  Best average reward: {results['best_avg_reward']:.2f}")
    print(f"  Converged: {'✅ Yes' if results['converged'] else '❌ No'}")
    
    if results['converged']:
        print("\n🎉 Congratulations! The agent successfully learned to balance the pole!")
    else:
        print("\n💪 Training completed. More episodes might be needed for full convergence.")
    
    print("\n🏁 Demo completed!")


if __name__ == "__main__":
    main()