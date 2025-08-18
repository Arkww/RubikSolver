import time
from dqn_agent import DQNAgent
from cube import Cube
from RL_env import RubiksCubeEnv
import numpy as np
from collections import deque


def complete_training(agent, env, cube_type="2x2"):
    """Progressive training: start with 1-move scrambles, increase difficulty when success rate > 90%"""
    
    agent.get_network_info()
    print(f"\nProgressive training on {cube_type} cube")
    
    max_scramble_moves = 20
    min_episodes_per_level = 100  # Minimum episodes before checking success rate
    success_threshold = 0.95  # 95% success rate to advance

    overall_start_time = time.time()
    
    for scramble_level in range(1, max_scramble_moves + 1):
        print(f"\n--- Training Level {scramble_level}: {scramble_level}-move scrambles ---")
        
        level_successes = 0
        level_episodes = 0
        level_rewards = []
        level_start_time = time.time()
        
        # Keep training at this level until we achieve good success rate
        while True:
            # Reset and scramble cube
            state = env.reset()
            env.scramble(scramble_level)
            state = env._get_observation()
            
            total_reward = 0
            steps = 0
            max_steps_per_episode = 50  # Prevent infinite episodes
            
            # Run one episode
            while steps < max_steps_per_episode:
                action = agent.act(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                # Store experience in replay buffer
                agent.remember(state, action, reward, next_state, done)
                
                # Train network periodically
                if len(agent.replay_buffer) > agent.batch_size and steps % 4 == 0:
                    agent.replay()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    if info.get('solved', False):
                        level_successes += 1
                    break
            
            level_episodes += 1
            level_rewards.append(total_reward)
            
            # Check if we should advance to next level
            if level_episodes >= min_episodes_per_level:
                current_success_rate = level_successes / level_episodes

                # Print progress every 500 episodes
                if level_episodes % 500 == 0:
                    elapsed_time = time.time() - level_start_time
                    avg_reward = np.mean(level_rewards[-500:])  # Last 500 episodes
                    print(f"  Episodes {level_episodes}: Success {current_success_rate:.1%}, "
                          f"Avg Reward {avg_reward:.2f}, ε={agent.epsilon:.3f}, "
                          f"Time {elapsed_time/60:.1f}min")
                
                # Check if we can advance
                if current_success_rate >= success_threshold:
                    elapsed_time = time.time() - level_start_time
                    print(f"  ✓ Level {scramble_level} completed! "
                          f"Success rate: {current_success_rate:.1%} "
                          f"({level_episodes} episodes, {elapsed_time/60:.1f}min)")
                    break
                
                # Safety check - don't train forever on one level
                if level_episodes >= 1000:
                    current_success_rate = level_successes / level_episodes
                    print(f"  ⚠ Max episodes reached for level {scramble_level}. "
                          f"Success rate: {current_success_rate:.1%}")
                    break
    
    total_time = time.time() - overall_start_time
    print(f"\n🎉 Training completed! Total time: {total_time/3600:.2f} hours")
    
    # Return final statistics
    return {
        'max_level_reached': scramble_level,
        'total_time_hours': total_time/3600,
        'final_epsilon': agent.epsilon
    }


def evaluate_agent(agent, env, scramble_moves, num_tests=100):
    """Evaluate agent performance on specific scramble level"""
    successes = 0
    total_steps = 0
    
    print(f"\nEvaluating on {scramble_moves}-move scrambles ({num_tests} tests)...")
    
    for test in range(num_tests):
        state = env.reset()
        env.scramble(scramble_moves)
        state = env._get_observation()
        
        steps = 0
        max_steps = 100
        
        while steps < max_steps:
            action = agent.act(state, training=False)  # No exploration during evaluation
            state, reward, done, info = env.step(action)
            steps += 1
            
            if done:
                if info.get('solved', False):
                    successes += 1
                total_steps += steps
                break
    
    success_rate = successes / num_tests
    avg_steps = total_steps / successes if successes > 0 else 0
    
    print(f"Results: {success_rate:.1%} success rate, "
          f"avg {avg_steps:.1f} steps when solved")
    
    return success_rate, avg_steps



def train_phase(agent, env ,scramble_moves, episodes, type): 
        """Training an agent for a set number of episodes and scrambled moves"""
        

        agent.get_network_info()
        print(f"\nTraining on {scramble_moves}-move scrambles for {episodes} episodes for a {type} cube")
        
        successes = 0
        episode_rewards = []
        recent_successes = 0
        start_time = time.time()
        
        for episode in range(episodes):
            state = env.reset()           # Start with solved cube
            env.scramble(scramble_moves) 
            state = env._get_observation() # Get scrambled state
            
            total_reward = 0
            steps = 0
            max_steps = max(100, scramble_moves * 5)  # More generous step limit
            
            while steps < max_steps:
                action = agent.act(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Train network (but not every step to speed up training)
                if len(agent.replay_buffer) > agent.batch_size and steps % 4 == 0:
                    agent.replay()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if info.get('solved', False):
                    successes += 1
                    recent_successes += 1
                    break
                    
                if done:
                    break
            
            episode_rewards.append(total_reward)
            
            # Progress reporting
            if (episode + 1) % 50 == 0:
                current_success_rate = successes / (episode + 1)
                recent_success_rate = recent_successes / 50
                avg_reward = np.mean(episode_rewards[-50:])
                elapsed_time = time.time() - start_time
                
                print(f"   Episode {episode+1:3d}/{episodes}: "
                    f"Success {current_success_rate:.1%} "
                    f"(recent {recent_success_rate:.1%}), "
                    f"Reward {avg_reward:6.1f}, "
                    f"ε={agent.epsilon:.3f}, "
                    f"Time {elapsed_time/60:.1f}min")
                
                recent_successes = 0
        
        final_success_rate = successes / episodes
        total_time = time.time() - start_time
        
        print(f"\nPhase Complete!")
        print(f"   Total Successes: {successes}/{episodes}")
        print(f"   Training Time: {total_time/60:.1f} minutes")

        return final_success_rate

def progressive_training():
        """Train with progressively harder scrambles"""
                
        phases = [
            (3, 30000),   # 1-move scrambles first
            (4, 1000),   # 2-move scrambles  
            (8, 1000),   # 3-move scrambles
            (10, 1000),   # 5-move scrambles
            #(10, 1000), # 10-move scrambles
           # (20, 2000)  # Finally 20-move scrambles
        ]

        env = RubiksCubeEnv(Cube, type='2x2')

        agent = DQNAgent(
                learning_rate=0.0005,        # Slower learning rate
                epsilon=0.9,                 # Higher initial exploration
                epsilon_decay=0.9995,        # Much slower decay
                epsilon_min=0.05,            # Higher minimum exploration
                gamma=0.95,                  # Slightly less future-focused
                batch_size=64,               # Larger batches for stability
                target_update_freq=1000,     # Less frequent target updates
                type="2x2"
            )
        
        
        for scramble_moves, episodes in phases:
            print(f"\n{'='*50}")
            print(f"PHASE: {scramble_moves}-move scrambles")
            success_rate = train_phase(agent,env, scramble_moves, episodes, "2x2")
            if success_rate < 0.5 :  
                return

