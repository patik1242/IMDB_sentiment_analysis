from clean_data import load_clean_data
from model_training import training_model

def main():

    clean_training = load_clean_data()
    training_model(clean_training)

if __name__ == "__main__":
    main()