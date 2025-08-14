import random


class Cube:
    def __init__(self, type="3x3"):
        if type == "3x3":
            self.type = "3x3"
            self.faces = {
                'U': [['W'] * 3 for _ in range(3)],  # White
                'D': [['Y'] * 3 for _ in range(3)],  # Yellow
                'F': [['G'] * 3 for _ in range(3)],  # Green
                'B': [['B'] * 3 for _ in range(3)],  # Blue
                'L': [['O'] * 3 for _ in range(3)],  # Orange
                'R': [['R'] * 3 for _ in range(3)],  # Red
            }
        elif type == "2x2":
            self.type = "2x2"
            self.faces = {
                'U': [['W'] * 2 for _ in range(2)],
                'D': [['Y'] * 2 for _ in range(2)],
                'F': [['G'] * 2 for _ in range(2)],
                'B': [['B'] * 2 for _ in range(2)],
                'L': [['O'] * 2 for _ in range(2)],
                'R': [['R'] * 2 for _ in range(2)],
            }
        else:
            raise ValueError("Unsupported cube type. Use '3x3' or '2x2'.")

    def rotate_face_cw(self, face):
        # Rotate a face 90 degrees clockwise
        self.faces[face] = [list(row) for row in zip(*self.faces[face][::-1])]

    def rotate_face_ccw(self, face):
        # Rotate a face 90 degrees counter-clockwise
        self.faces[face] = [list(row) for row in zip(*self.faces[face])][::-1]

    def scramble(self, rate=50):
        for _ in range(rate):  # Perform random moves
            self.move(random.choice(['U', 'D', 'L', 'R', 'F', 'B']))
        pass

    def is_solved(self):
        # Check if all faces are solved (all stickers of the same color)
        return all(all(sticker == face[0][0] for sticker in row) for face in self.faces.values() for row in face)
    
    def reset_to_solved(self):
        # Reset the cube to the solved state
        if self.type == '3x3':
            self.faces = {
                'U': [['W'] * 3 for _ in range(3)],
                'D': [['Y'] * 3 for _ in range(3)],
                'F': [['G'] * 3 for _ in range(3)],
                'B': [['B'] * 3 for _ in range(3)],
                'L': [['O'] * 3 for _ in range(3)],
                'R': [['R'] * 3 for _ in range(3)],
            }
        elif self.type == '2x2':
            self.faces = {
                'U': [['W'] * 2 for _ in range(2)],
                'D': [['Y'] * 2 for _ in range(2)],
                'F': [['G'] * 2 for _ in range(2)],
                'B': [['B'] * 2 for _ in range(2)],
                'L': [['O'] * 2 for _ in range(2)],
                'R': [['R'] * 2 for _ in range(2)],
            }


    def move(self, notation):
        if self.type == '3x3':
            # Your 3x3 code looks correct, keeping it as is
            match(notation):
                case 'U':
                    self.rotate_face_cw('U')
                    temp = self.faces['F'][0]
                    self.faces['F'][0] = self.faces['R'][0]
                    self.faces['R'][0] = self.faces['B'][0]
                    self.faces['B'][0] = self.faces['L'][0]
                    self.faces['L'][0] = temp
                case 'U\'':
                    self.rotate_face_ccw('U')
                    temp = self.faces['F'][0]
                    self.faces['F'][0] = self.faces['L'][0]
                    self.faces['L'][0] = self.faces['B'][0]
                    self.faces['B'][0] = self.faces['R'][0]
                    self.faces['R'][0] = temp
                case 'D':
                    self.rotate_face_cw('D')
                    temp = self.faces['F'][2]
                    self.faces['F'][2] = self.faces['R'][2]
                    self.faces['R'][2] = self.faces['B'][2]
                    self.faces['B'][2] = self.faces['L'][2]
                    self.faces['L'][2] = temp
                case 'D\'':
                    self.rotate_face_ccw('D')
                    temp = self.faces['F'][2]
                    self.faces['F'][2] = self.faces['L'][2]
                    self.faces['L'][2] = self.faces['B'][2]
                    self.faces['B'][2] = self.faces['R'][2]
                    self.faces['R'][2] = temp
                case 'F':
                    self.rotate_face_cw('F')
                    temp = [self.faces['U'][2][0], self.faces['U'][2][1], self.faces['U'][2][2]]
                    self.faces['U'][2][0] = self.faces['L'][2][2]
                    self.faces['U'][2][1] = self.faces['L'][1][2]
                    self.faces['U'][2][2] = self.faces['L'][0][2]
                    self.faces['L'][2][2] = self.faces['D'][0][0]
                    self.faces['L'][1][2] = self.faces['D'][0][1]
                    self.faces['L'][0][2] = self.faces['D'][0][2]
                    self.faces['D'][0][0] = self.faces['R'][0][0]
                    self.faces['D'][0][1] = self.faces['R'][1][0]
                    self.faces['D'][0][2] = self.faces['R'][2][0]
                    self.faces['R'][0][0] = temp[0]
                    self.faces['R'][1][0] = temp[1]
                    self.faces['R'][2][0] = temp[2]
                case 'F\'':
                    self.rotate_face_ccw('F')   
                    temp = [self.faces['U'][2][0], self.faces['U'][2][1], self.faces['U'][2][2]]
                    self.faces['U'][2][0] = self.faces['R'][0][0]
                    self.faces['U'][2][1] = self.faces['R'][1][0]
                    self.faces['U'][2][2] = self.faces['R'][2][0]
                    self.faces['R'][0][0] = self.faces['D'][0][0]
                    self.faces['R'][1][0] = self.faces['D'][0][1]
                    self.faces['R'][2][0] = self.faces['D'][0][2]
                    self.faces['D'][0][0] = self.faces['L'][2][2]
                    self.faces['D'][0][1] = self.faces['L'][1][2]
                    self.faces['D'][0][2] = self.faces['L'][0][2]
                    self.faces['L'][2][2] = temp[0]
                    self.faces['L'][1][2] = temp[1]
                    self.faces['L'][0][2] = temp[2]
                case 'B':
                    self.rotate_face_cw('B')
                    temp = [self.faces['U'][0][0], self.faces['U'][0][1], self.faces['U'][0][2]]
                    self.faces['U'][0][0] = self.faces['R'][0][2]
                    self.faces['U'][0][1] = self.faces['R'][1][2]
                    self.faces['U'][0][2] = self.faces['R'][2][2]
                    self.faces['R'][0][2] = self.faces['D'][2][2]
                    self.faces['R'][1][2] = self.faces['D'][2][1]
                    self.faces['R'][2][2] = self.faces['D'][2][0]
                    self.faces['D'][2][2] = self.faces['L'][2][0]
                    self.faces['D'][2][1] = self.faces['L'][1][0]
                    self.faces['D'][2][0] = self.faces['L'][0][0]
                    self.faces['L'][2][0] = temp[0]
                    self.faces['L'][1][0] = temp[1]
                    self.faces['L'][0][0] = temp[2]
                case 'B\'':
                    self.rotate_face_ccw('B')
                    temp = [self.faces['U'][0][0], self.faces['U'][0][1], self.faces['U'][0][2]]
                    self.faces['U'][0][0] = self.faces['L'][2][0]
                    self.faces['U'][0][1] = self.faces['L'][1][0]
                    self.faces['U'][0][2] = self.faces['L'][0][0]
                    self.faces['L'][2][0] = self.faces['D'][2][2]
                    self.faces['L'][1][0] = self.faces['D'][2][1]
                    self.faces['L'][0][0] = self.faces['D'][2][0]
                    self.faces['D'][2][2] = self.faces['R'][0][2]
                    self.faces['D'][2][1] = self.faces['R'][1][2]
                    self.faces['D'][2][0] = self.faces['R'][2][2]
                    self.faces['R'][0][2] = temp[0]
                    self.faces['R'][1][2] = temp[1]
                    self.faces['R'][2][2] = temp[2]
                case 'L':
                    self.rotate_face_cw('L') 
                    temp = [self.faces['F'][0][0], self.faces['F'][1][0], self.faces['F'][2][0]]
                    self.faces['F'][0][0] = self.faces['U'][0][0]
                    self.faces['F'][1][0] = self.faces['U'][1][0] 
                    self.faces['F'][2][0] = self.faces['U'][2][0]
                    self.faces['U'][0][0] = self.faces['B'][2][2] 
                    self.faces['U'][1][0] = self.faces['B'][1][2]
                    self.faces['U'][2][0] = self.faces['B'][0][2]
                    self.faces['B'][2][2] = self.faces['D'][0][0]
                    self.faces['B'][1][2] = self.faces['D'][1][0]
                    self.faces['B'][0][2] = self.faces['D'][2][0]
                    self.faces['D'][0][0] = temp[0]
                    self.faces['D'][1][0] = temp[1]
                    self.faces['D'][2][0] = temp[2]
                case 'L\'':
                    self.rotate_face_ccw('L')
                    temp = [self.faces['F'][0][0], self.faces['F'][1][0], self.faces['F'][2][0]]
                    self.faces['F'][0][0] = self.faces['D'][0][0]
                    self.faces['F'][1][0] = self.faces['D'][1][0]
                    self.faces['F'][2][0] = self.faces['D'][2][0]
                    self.faces['D'][0][0] = self.faces['B'][2][2]
                    self.faces['D'][1][0] = self.faces['B'][1][2]
                    self.faces['D'][2][0] = self.faces['B'][0][2]
                    self.faces['B'][2][2] = self.faces['U'][0][0]
                    self.faces['B'][1][2] = self.faces['U'][1][0]
                    self.faces['B'][0][2] = self.faces['U'][2][0]
                    self.faces['U'][0][0] = temp[0]
                    self.faces['U'][1][0] = temp[1]
                    self.faces['U'][2][0] = temp[2]
                case 'R':
                    self.rotate_face_cw('R')
                    temp = [self.faces['F'][0][2], self.faces['F'][1][2], self.faces['F'][2][2]]  
                    self.faces['F'][0][2] = self.faces['D'][0][2] 
                    self.faces['F'][1][2] = self.faces['D'][1][2]
                    self.faces['F'][2][2] = self.faces['D'][2][2]
                    self.faces['D'][0][2] = self.faces['B'][2][0] 
                    self.faces['D'][1][2] = self.faces['B'][1][0]
                    self.faces['D'][2][2] = self.faces['B'][0][0]
                    self.faces['B'][2][0] = self.faces['U'][0][2]  
                    self.faces['B'][1][0] = self.faces['U'][1][2]
                    self.faces['B'][0][0] = self.faces['U'][2][2]
                    self.faces['U'][0][2] = temp[0]
                    self.faces['U'][1][2] = temp[1]
                    self.faces['U'][2][2] = temp[2]
                case 'R\'':
                    self.rotate_face_ccw('R')
                    temp = [self.faces['F'][0][2], self.faces['F'][1][2], self.faces['F'][2][2]]
                    self.faces['F'][0][2] = self.faces['U'][0][2]
                    self.faces['F'][1][2] = self.faces['U'][1][2]
                    self.faces['F'][2][2] = self.faces['U'][2][2]
                    self.faces['U'][0][2] = self.faces['B'][2][0]
                    self.faces['U'][1][2] = self.faces['B'][1][0]
                    self.faces['U'][2][2] = self.faces['B'][0][0]
                    self.faces['B'][2][0] = self.faces['D'][0][2]
                    self.faces['B'][1][0] = self.faces['D'][1][2]
                    self.faces['B'][0][0] = self.faces['D'][2][2]
                    self.faces['D'][0][2] = temp[0]
                    self.faces['D'][1][2] = temp[1]
                    self.faces['D'][2][2] = temp[2]
                case _:
                    pass
                    
        elif self.type == '2x2':
            # FIXED 2x2 IMPLEMENTATION
            match(notation):
                case 'U':
                    self.rotate_face_cw('U')
                    # Move top edges: F[0] -> R[0] -> B[0] -> L[0] -> F[0]
                    temp = self.faces['F'][0][:]  # Copy entire row
                    self.faces['F'][0] = self.faces['R'][0][:]
                    self.faces['R'][0] = self.faces['B'][0][:]
                    self.faces['B'][0] = self.faces['L'][0][:]
                    self.faces['L'][0] = temp
                    
                case 'U\'':
                    self.rotate_face_ccw('U')
                    # Reverse direction: F[0] -> L[0] -> B[0] -> R[0] -> F[0]
                    temp = self.faces['F'][0][:]
                    self.faces['F'][0] = self.faces['L'][0][:]
                    self.faces['L'][0] = self.faces['B'][0][:]
                    self.faces['B'][0] = self.faces['R'][0][:]
                    self.faces['R'][0] = temp
                    
                case 'D':
                    self.rotate_face_cw('D')
                    # Move bottom edges: F[1] -> R[1] -> B[1] -> L[1] -> F[1]
                    temp = self.faces['F'][1][:]
                    self.faces['F'][1] = self.faces['R'][1][:]
                    self.faces['R'][1] = self.faces['B'][1][:]
                    self.faces['B'][1] = self.faces['L'][1][:]
                    self.faces['L'][1] = temp
                    
                case 'D\'':
                    self.rotate_face_ccw('D')
                    # Reverse direction: F[1] -> L[1] -> B[1] -> R[1] -> F[1]
                    temp = self.faces['F'][1][:]
                    self.faces['F'][1] = self.faces['L'][1][:]
                    self.faces['L'][1] = self.faces['B'][1][:]
                    self.faces['B'][1] = self.faces['R'][1][:]
                    self.faces['R'][1] = temp
                    
                case 'F':
                    self.rotate_face_cw('F')
                    # Cycle: U[1] -> L[right_col] -> D[0] -> R[left_col] -> U[1]
                    temp = [self.faces['U'][1][0], self.faces['U'][1][1]]
                    self.faces['U'][1][0] = self.faces['L'][1][1]  # L bottom-right
                    self.faces['U'][1][1] = self.faces['L'][0][1]  # L top-right
                    self.faces['L'][1][1] = self.faces['D'][0][1]  # D top-right
                    self.faces['L'][0][1] = self.faces['D'][0][0]  # D top-left
                    self.faces['D'][0][1] = self.faces['R'][0][0]  # R top-left
                    self.faces['D'][0][0] = self.faces['R'][1][0]  # R bottom-left
                    self.faces['R'][0][0] = temp[1]  # U bottom-right
                    self.faces['R'][1][0] = temp[0]  # U bottom-left
                    
                case 'F\'':
                    self.rotate_face_ccw('F')
                    # Reverse cycle: U[1] -> R[left_col] -> D[0] -> L[right_col] -> U[1]
                    temp = [self.faces['U'][1][0], self.faces['U'][1][1]]
                    self.faces['U'][1][0] = self.faces['R'][1][0]  # R bottom-left
                    self.faces['U'][1][1] = self.faces['R'][0][0]  # R top-left
                    self.faces['R'][1][0] = self.faces['D'][0][0]  # D top-left
                    self.faces['R'][0][0] = self.faces['D'][0][1]  # D top-right
                    self.faces['D'][0][0] = self.faces['L'][0][1]  # L top-right
                    self.faces['D'][0][1] = self.faces['L'][1][1]  # L bottom-right
                    self.faces['L'][0][1] = temp[1]  # U bottom-right
                    self.faces['L'][1][1] = temp[0]  # U bottom-left
                    
                case 'B':
                    self.rotate_face_cw('B')
                    # Cycle: U[0] -> R[right_col] -> D[1] -> L[left_col] -> U[0]
                    temp = [self.faces['U'][0][0], self.faces['U'][0][1]]
                    self.faces['U'][0][0] = self.faces['R'][1][1]  # R bottom-right
                    self.faces['U'][0][1] = self.faces['R'][0][1]  # R top-right
                    self.faces['R'][1][1] = self.faces['D'][1][1]  # D bottom-right
                    self.faces['R'][0][1] = self.faces['D'][1][0]  # D bottom-left
                    self.faces['D'][1][1] = self.faces['L'][0][0]  # L top-left
                    self.faces['D'][1][0] = self.faces['L'][1][0]  # L bottom-left
                    self.faces['L'][0][0] = temp[1]  # U top-right
                    self.faces['L'][1][0] = temp[0]  # U top-left
                    
                case 'B\'':
                    self.rotate_face_ccw('B')
                    # Reverse cycle: U[0] -> L[left_col] -> D[1] -> R[right_col] -> U[0]
                    temp = [self.faces['U'][0][0], self.faces['U'][0][1]]
                    self.faces['U'][0][0] = self.faces['L'][1][0]  # L bottom-left
                    self.faces['U'][0][1] = self.faces['L'][0][0]  # L top-left
                    self.faces['L'][1][0] = self.faces['D'][1][0]  # D bottom-left
                    self.faces['L'][0][0] = self.faces['D'][1][1]  # D bottom-right
                    self.faces['D'][1][0] = self.faces['R'][0][1]  # R top-right
                    self.faces['D'][1][1] = self.faces['R'][1][1]  # R bottom-right
                    self.faces['R'][0][1] = temp[1]  # U top-right
                    self.faces['R'][1][1] = temp[0]  # U top-left
                    
                case 'L':
                    self.rotate_face_cw('L')
                    # Cycle: U[left_col] -> F[left_col] -> D[left_col] -> B[right_col] -> U[left_col]
                    temp = [self.faces['U'][0][0], self.faces['U'][1][0]]
                    self.faces['U'][0][0] = self.faces['B'][1][1]  # B bottom-right
                    self.faces['U'][1][0] = self.faces['B'][0][1]  # B top-right
                    self.faces['B'][1][1] = self.faces['D'][1][0]  # D bottom-left
                    self.faces['B'][0][1] = self.faces['D'][0][0]  # D top-left
                    self.faces['D'][1][0] = self.faces['F'][1][0]  # F bottom-left
                    self.faces['D'][0][0] = self.faces['F'][0][0]  # F top-left
                    self.faces['F'][1][0] = temp[1]  # U bottom-left
                    self.faces['F'][0][0] = temp[0]  # U top-left
                    
                case 'L\'':
                    self.rotate_face_ccw('L')
                    # Reverse cycle: U[left_col] -> B[right_col] -> D[left_col] -> F[left_col] -> U[left_col]
                    temp = [self.faces['U'][0][0], self.faces['U'][1][0]]
                    self.faces['U'][0][0] = self.faces['F'][0][0]  # F top-left
                    self.faces['U'][1][0] = self.faces['F'][1][0]  # F bottom-left
                    self.faces['F'][0][0] = self.faces['D'][0][0]  # D top-left
                    self.faces['F'][1][0] = self.faces['D'][1][0]  # D bottom-left
                    self.faces['D'][0][0] = self.faces['B'][0][1]  # B top-right
                    self.faces['D'][1][0] = self.faces['B'][1][1]  # B bottom-right
                    self.faces['B'][0][1] = temp[1]  # U bottom-left
                    self.faces['B'][1][1] = temp[0]  # U top-left
                    
                case 'R':
                    self.rotate_face_cw('R')
                    # Cycle: U[right_col] -> B[left_col] -> D[right_col] -> F[right_col] -> U[right_col]
                    temp = [self.faces['U'][0][1], self.faces['U'][1][1]]
                    self.faces['U'][0][1] = self.faces['F'][0][1]  # F top-right
                    self.faces['U'][1][1] = self.faces['F'][1][1]  # F bottom-right
                    self.faces['F'][0][1] = self.faces['D'][0][1]  # D top-right
                    self.faces['F'][1][1] = self.faces['D'][1][1]  # D bottom-right
                    self.faces['D'][0][1] = self.faces['B'][1][0]  # B bottom-left
                    self.faces['D'][1][1] = self.faces['B'][0][0]  # B top-left
                    self.faces['B'][1][0] = temp[1]  # U bottom-right
                    self.faces['B'][0][0] = temp[0]  # U top-right
                    
                case 'R\'':
                    self.rotate_face_ccw('R')
                    # Reverse cycle: U[right_col] -> F[right_col] -> D[right_col] -> B[left_col] -> U[right_col]
                    temp = [self.faces['U'][0][1], self.faces['U'][1][1]]
                    self.faces['U'][0][1] = self.faces['B'][0][0]  # B top-left
                    self.faces['U'][1][1] = self.faces['B'][1][0]  # B bottom-left
                    self.faces['B'][0][0] = self.faces['D'][1][1]  # D bottom-right
                    self.faces['B'][1][0] = self.faces['D'][0][1]  # D top-right
                    self.faces['D'][1][1] = self.faces['F'][1][1]  # F bottom-right
                    self.faces['D'][0][1] = self.faces['F'][0][1]  # F top-right
                    self.faces['F'][1][1] = temp[1]  # U bottom-right
                    self.faces['F'][0][1] = temp[0]  # U top-right
                    
                case _:
                    pass


    # Visual representation functions, courtesy of ClaudeAI

    def __str__(self):
        """Enhanced string representation with better formatting"""
        result = ""
        for face in ['U', 'D', 'F', 'B', 'L', 'R']:
            result += f"╭─── {face} Face ───╮\n"
            for row in self.faces[face]:
                result += "│ " + " ".join(f"[{cell}]" for cell in row) + " │\n"
            result += "╰─────────────╯\n\n"
        return result


    def display_cube(self):
        """Use Unicode symbols for better visualization"""
        symbols = {
            'W': '⬜', 'Y': '🟨', 'R': '🟥', 
            'O': '🟧', 'G': '🟩', 'B': '🟦'
        }
        
        result = "        ┌─────┐\n"
        result += "        │  U  │\n"
        for row in self.faces['U']:
            result += "        │ " + "".join(symbols.get(cell, cell) for cell in row) + " │\n"
        result += "  ┌─────┼─────┼─────┐\n"
        result += "  │  L  │  F  │  R  │\n"
        
        for i in range(len(self.faces['L'])):
            l_symbols = "".join(symbols.get(cell, cell) for cell in self.faces['L'][i])
            f_symbols = "".join(symbols.get(cell, cell) for cell in self.faces['F'][i])
            r_symbols = "".join(symbols.get(cell, cell) for cell in self.faces['R'][i])
            result += f"  │ {l_symbols} │ {f_symbols} │ {r_symbols} │\n"
        
        result += "  └─────┼─────┼─────┘\n"
        result += "        │  D  │\n"
        for row in self.faces['D']:
            result += "        │ " + "".join(symbols.get(cell, cell) for cell in row) + " │\n"
        result += "        └─────┘\n"
        result += "        ┌─────┐\n"
        result += "        │  B  │\n"
        for row in self.faces['B']:
            result += "        │ " + "".join(symbols.get(cell, cell) for cell in row) + " │\n"
        result += "        └─────┘\n"
        
        return result
