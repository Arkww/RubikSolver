from cube import Cube

def main():
    cube = Cube()
    print(cube.display_cube())
    cube.move('U')
    print(cube.display_cube())
    cube.move('R')
    print(cube.display_cube())
    cube.scramble()
    print(cube.display_cube())


if __name__ == "__main__":
    main()