from config import setup_monitor
from experiment import run_experiment

def main():
    mon = setup_monitor()
    run_experiment(mon)

if __name__ == "__main__":
    main()