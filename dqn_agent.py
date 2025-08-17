import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from replay_buffer import ReplayBuffer
from cube_dqn import CubeDQN

class DQNAgent:
    """
    Complete DQN Agent with neural network
    This contains the neural network and all the learning logic
    """
    
    def __init__(
        self,
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        gamma=0.99,
        batch_size=32,
        buffer_size=100000,
        target_update_freq=500,
        device=None,
        type=None
    ):
        
        self.action_size = 12  # 6 faces * 2 directions

        # Set cube-specific parameters based on type
        if type == "3x3":
            self.state_size = 54  # 6 faces * 9 stickers
        elif type == "2x2":
            self.state_size = 24  # 6 faces * 4 stickers
        else:
            raise ValueError("type must be '3x3' or '2x2'")
                
        # Store parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.training_step = 0
        self.type = type
        
        # Set device (GPU if available, otherwise CPU)
    
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'npu')
        print(f"   Using device: {self.device}")
        
        # CREATE THE NEURAL NETWORKS

        if self.type == "3x3":
            # Main network: used for choosing actions and learning
            # Target network: provides stable learning targets
            self.q_network = CubeDQN("3x3", [512, 256, 128], self.action_size).to(self.device)
            self.target_network = CubeDQN("3x3", [512, 256, 128], self.action_size).to(self.device)
        elif self.type == "2x2":
            # Use 3 hidden layers for 2x2 as well, but smaller sizes
            self.q_network = CubeDQN("2x2", [256, 128, 64], self.action_size).to(self.device)
            self.target_network = CubeDQN("2x2", [256, 128, 64], self.action_size).to(self.device)

        # Copy weights from main to target network
        self.update_target_network()
        
        # Optimizer for training the network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.losses = []
        self.episode_rewards = []
        
        print(f"Network parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
    
    def act(self, state, training=True):
        """
        Choose action using the neural network
        """
        
        # Epsilon-greedy policy: sometimes explore, sometimes exploit
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_size - 1)
        
        # Exploit: use neural network to choose best action
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values from neural network
        with torch.no_grad():  # Don't compute gradients for inference
            q_values = self.q_network(state_tensor)
        
        # Choose action with highest Q-value
        best_action = q_values.argmax().item()
        return best_action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def replay(self):
        """
        Train the neural network on past experiences
        """
        
        # Need enough experiences to train
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample random batch of experiences
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Convert experiences to tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.BoolTensor(np.array([e.done for e in experiences])).to(self.device)

        # Current Q-values: Q(s,a) for actions taken
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values: r + γ * max(Q(s',a'))
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss between current and target Q-values
        loss = self.loss_function(current_q_values.squeeze(), target_q_values)
        
        # NEURAL NETWORK LEARNING STEP
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()             # Compute gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)  # Prevent exploding gradients
        self.optimizer.step()       # Update network weights
        
        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        # Store loss for monitoring
        self.losses.append(loss.item())
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """Save the trained neural network"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'losses': self.losses,
            'episode_rewards': self.episode_rewards,
            'type': self.type,
            'state_size': self.state_size,
            'action_size': self.action_size
        }, filepath)
        print(f"Neural network saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained neural network"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.losses = checkpoint.get('losses', [])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        print(f"Neural network loaded from {filepath}")
    
    def get_network_info(self):
        """Get information about the neural network"""
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)

        print(f"\nNeural Network Information:")
        if self.type == "3x3":
            print(f"Architecture: 54 → 512 → 256 → 128 → 12")
        else:
            print(f"Architecture: 24 → 256 → 128 → 6")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Device: {self.device}")
        print(f"Memory usage: ~{total_params * 4 / 1024 / 1024:.1f} MB")