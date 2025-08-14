import torch
import torch.nn as nn
from collections import deque, namedtuple


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class CubeDQN(nn.Module):
    """
    Deep Q-Network for Rubik's Cube solving
    This IS the neural network that learns to solve the cube
    """
    
    def __init__(self, type='3x3', hidden_sizes=[512, 256, 128], output_size=None):
        """
        Args:
            type (str): '3x3' or '2x2'
            hidden_sizes (list): Sizes of hidden layers
            output_size (int): Number of possible actions (moves)
        """
        super(CubeDQN, self).__init__()

        # Set input size and output size based on cube type
        if type == '3x3':
            input_size = 54  # 6 faces * 9 stickers
            if output_size is None:
                output_size = 12  # 6 faces * 2 directions
        elif type == '2x2':
            input_size = 24  # 6 faces * 4 stickers
            if output_size is None:
                output_size = 12 
        else:
            raise ValueError("type must be '3x3' or '2x2'")

        print(f"   Creating Neural Network:")
        print(f"   Cube type: {type}")
        print(f"   Input: {input_size} (cube state)")
        print(f"   Hidden layers: {hidden_sizes}")
        print(f"   Output: {output_size} (Q-values for each move)")

        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.dropout1 = nn.Dropout(0.2)

        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.dropout2 = nn.Dropout(0.2)

        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.dropout3 = nn.Dropout(0.2)

        self.output_layer = nn.Linear(hidden_sizes[2], output_size)

        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for better learning"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x):
        """
        Forward pass: cube state → Q-values
        """
        # Ensure input is float tensor
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        # Layer 1: Basic pattern recognition
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        
        # Layer 2: Intermediate strategy
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        
        # Layer 3: High-level planning
        x = torch.relu(self.layer3(x))
        x = self.dropout3(x)
        
        # Output: Q-values for each action (no activation)
        q_values = self.output_layer(x)
        
        return q_values

