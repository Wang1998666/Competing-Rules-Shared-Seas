"""Rebuild the map panels from the archived tables; outputs stay in this folder."""
from draw_candidates import FUNCTIONS, ORDER, m

def main():
    m.MAIN.mkdir(exist_ok=True)
    m.SI.mkdir(exist_ok=True)
    m.load_data()
    for figure in ORDER:
        FUNCTIONS[figure]()

if __name__ == "__main__":
    main()
