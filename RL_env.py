
from typing import Tuple, Dict
import gym
from gym import spaces
from cube import Cube
import numpy as np
from dqn_agent import DQNAgent
import time

class RubiksCubeEnv(gym.Env):
    def __init__(self, cube_class, type="3x3"):
        super(RubiksCubeEnv, self).__init__()
        
        self.cube_class = cube_class
        self.cube = None
        self.type = type

        # Define action space
        if type == "3x3":
            self.action_space = spaces.Discrete(12)
            obs_size = 54  # 6 faces * 9 stickers
        elif type == "2x2":
            self.action_space = spaces.Discrete(6)   # Only 6 moves for 2x2
            obs_size = 24  # 6 faces * 4 stickers
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Rest of initialization...
        self.color_to_number = {'W': 0, 'R': 1, 'B': 2, 'O': 3, 'G': 4, 'Y': 5}
        self.action_map = {
            0: 'F',  1: 'R',  2: 'U',  3: 'L',  4: 'D',  5: 'B',
            6: "F'", 7: "R'", 8: "U'", 9: "L'", 10: "D'", 11: "B'"
        }
        self.moves_count = 0
        self.max_moves = 100
        self.debug = False
        
        # Store solved state for comparison
        self.solved_state = None
    
    def reset(self) -> np.ndarray:
        """Reset environment to solved state"""
        self.cube = self.cube_class(type=self.type)
        self.moves_count = 0
        
        # Store the solved state for comparison
        if self.solved_state is None:
            self.solved_state = self._get_observation().copy()
        
        return self._get_observation()
    
    def scramble(self, num_moves: int):
        """Scramble the cube - call this AFTER reset()"""
        scramble_actions = []
        for _ in range(num_moves):
            if self.type == "3x3":
                action = np.random.randint(0, 12)
            else:  # 2x2
                action = np.random.randint(0, 6)
            scramble_actions.append(action)
            self.cube.move(self.action_map[action])
        
        self.moves_count = 0  # Reset move counter after scrambling
        return scramble_actions

    def _count_correct_stickers(self) -> int:
        """Count how many stickers are in their correct positions"""
        if self.solved_state is None:
            return 0
            
        current_state = self._get_observation()
        correct_count = 0
        
        # Compare each position with solved state
        for i in range(len(current_state)):
            if abs(current_state[i] - self.solved_state[i]) < 0.001:  # Account for float precision
                correct_count += 1
        
        return correct_count

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
            """Execute action and return (observation, reward, done, info)"""
            
            if self.debug and self.moves_count < 5:  # Only debug first few moves
                print(f"Step {self.moves_count}: Action {action} ({self.action_map[action]})")
                print(f"  Before move - Solved: {self.cube.is_solved()}")
            
            # Execute the move
            move = self.action_map[action]
            self.cube.move(move)
            self.moves_count += 1
            
            # Get new state
            obs = self._get_observation()
            
            # Calculate reward
            reward = self._calculate_reward()
            
            # Check if episode is done
            is_solved = self.cube.is_solved()
            done = is_solved or self.moves_count >= self.max_moves
            
            if self.debug and self.moves_count < 5:
                print(f"  After move - Solved: {is_solved}, Reward: {reward}")
            
            # Additional info
            info = {
                'moves': self.moves_count,
                'solved': is_solved,
                'move_executed': move,
                'max_moves_reached': self.moves_count >= self.max_moves,
                'reward_breakdown': self._get_reward_breakdown()
            }
            
            return obs, reward, done, info
        
    def _get_observation(self) -> np.ndarray:
            """Convert cube state to normalized observation vector"""
            observation = []
            
            # Get faces in consistent order (important!)
            face_order = ['U', 'R', 'F', 'D', 'L', 'B']  # Standard order
            
            for face_name in face_order:
                if face_name not in self.cube.faces:
                    # Handle different face naming conventions
                    face_name = list(self.cube.faces.keys())[face_order.index(face_name)]
                    
                face_matrix = self.cube.faces[face_name]
                
                # Convert 3x3 face to flat list of normalized numbers
                for row in range(len(self.cube.faces[face_name])):
                    for col in range(len(self.cube.faces[face_name][row])):
                        color_char = face_matrix[row][col]
                        color_number = self.color_to_number[color_char]
                        # Normalize to 0-1 range
                        normalized_value = color_number / 5.0
                        observation.append(normalized_value)
            
            return np.array(observation, dtype=np.float32)
        
    def _calculate_reward(self) -> float:
            """Reward function"""
            
            if self.cube.is_solved():
                # LARGE positive reward for solving
                base_reward = 100.0
                
                # Bonus for solving efficiently
                efficiency_bonus = max(0, 20 - self.moves_count)
                
                total_reward = base_reward + efficiency_bonus
                
                if self.debug:
                    print(f"  SOLVED! Base: {base_reward}, Efficiency: {efficiency_bonus}, Total: {total_reward}")
                    
                return total_reward
            else:
                # Small penalty for each move + partial progress reward
                move_penalty = -0.1
                progress_reward = self._calculate_progress_reward()
                
                total_reward = move_penalty + progress_reward
                
                # Prevent extremely negative rewards
                total_reward = max(total_reward, -10.0)
                
                return total_reward
        
    def _calculate_progress_reward(self) -> float:
            """Calculate reward based on partial progress"""
            progress = 0.0
            
            # Method 1: Count solved stickers in correct positions
            solved_stickers = self._count_correct_stickers()
            progress += solved_stickers * 0.1  # Small reward per correct sticker
            
            # Method 2: Count completely solved faces
            solved_faces = self._count_solved_faces()
            progress += solved_faces * 2.0  # Bigger reward per solved face
            
            return min(progress, 10.0)  # Cap the progress reward
        

        
    def _count_solved_faces(self) -> int:
            """Count how many faces are completely solved"""
            solved_count = 0
            
            for face_name, face_matrix in self.cube.faces.items():
                # Check if all stickers on this face are the same color
                first_color = face_matrix[0][0]
                face_solved = True
                
                for i in range(len(self.cube.faces[face_name][0])):
                    for j in range(len(self.cube.faces[face_name][0])):
                        if face_matrix[i][j] != first_color:
                            face_solved = False
                            break
                    if not face_solved:
                        break
                
                if face_solved:
                    solved_count += 1
                    
            return solved_count
        
    def _get_reward_breakdown(self) -> Dict:
            """Get detailed reward breakdown for debugging"""
            if self.cube.is_solved():
                return {
                    'type': 'solved',
                    'base_reward': 100.0,
                    'efficiency_bonus': max(0, 20 - self.moves_count),
                    'total': self._calculate_reward()
                }
            else:
                return {
                    'type': 'in_progress',
                    'move_penalty': -0.1,
                    'progress_reward': self._calculate_progress_reward(),
                    'solved_faces': self._count_solved_faces(),
                    'correct_stickers': self._count_correct_stickers(),
                    'total': self._calculate_reward()
                }
        
    def get_simple_inverse_action(self, action: int) -> int:
            """Get the inverse of an action (approximate)"""
            return self.reverse_actions.get(action, action) 

