from clean_data import load_clean_data
from model import training_model

def main():
    print("Ładuje dane....")
    clean_training = load_clean_data()
    print("Zaczyna się trening...")
    training_model(clean_training)

if __name__ == "__main__":
    main()