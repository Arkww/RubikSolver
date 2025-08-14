import time
from dqn_agent import DQNAgent
from cube import Cube
from RL_env import RubiksCubeEnv
import numpy as np



def complete_training(agent, env, type): 
        """Training an agent for a set number of episodes and scrambled moves"""
        

        agent.get_network_info()
        print(f"\nTraining on all the scrambles on a {type} cube")
        
        episode_rewards = []
        start_time = time.time()
        
        for i in range(20,1,1):
            state = env.reset()           # Start with solved cube
            env.scramble(i) 
            state = env._get_observation() # Get scrambled state
            
            total_reward = 0
            succes_rate = 0
            steps = 0
            recent_successes = 0
            successes = 0


            while succes_rate < 0.9:
                steps =+ 1
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
                    
            
                episode_rewards.append(total_reward)
            
                # Progress reporting
                if (steps + 1) % 50 == 0:
                    current_success_rate = successes / (steps + 1)
                    recent_success_rate = recent_successes / 50
                    avg_reward = np.mean(episode_rewards[-50:])
                    elapsed_time = time.time() - start_time
                    
                    print(f"   Episode {steps+1:3d}/{steps}: "
                        f"Success {current_success_rate:.1%} "
                        f"(recent {recent_success_rate:.1%}), "
                        f"Reward {avg_reward:6.1f}, "
                        f"ε={agent.epsilon:.3f}, "
                        f"Time {elapsed_time/60:.1f}min")
                    
                print("Attained 0.9% accuracy consistenly")
                    
        
        total_time = time.time() - start_time
        

        return final_success_rate

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

