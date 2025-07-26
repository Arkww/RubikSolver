import random


class Cube:
    def __init__(self):
        # Faces: U (up), D (down), F (front), B (back), L (left), R (right)
        # Each face is a 3x3 matrix, initialized with a single color
        self.faces = {
            'U': [['W'] * 3 for _ in range(3)],  # White
            'D': [['Y'] * 3 for _ in range(3)],  # Yellow
            'F': [['G'] * 3 for _ in range(3)],  # Green
            'B': [['B'] * 3 for _ in range(3)],  # Blue
            'L': [['O'] * 3 for _ in range(3)],  # Orange
            'R': [['R'] * 3 for _ in range(3)],  # Red
        }

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

    def move(self, notation):
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
        
        for i in range(3):
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