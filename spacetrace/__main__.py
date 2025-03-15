import sys, os.path
from .scene import Scene
from .main import show_scene

HELP_MSG = "Usage: python -m spacetrace <path-to-scene>"

def main():
    if len(sys.argv) < 2:
        print(HELP_MSG)
        exit(1)

    path = sys.argv[1]
    if path.lower() in ('-h', '--help'):
        print(HELP_MSG)
        exit(1)

    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file found: {path}")
    scene = Scene.load(path)
    show_scene(scene)

if __name__ == '__main__':
    main()