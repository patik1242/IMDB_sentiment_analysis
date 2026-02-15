import pandas as pd
from preprocessing import preprocess_base

def load_clean_data():
    """
    Wczytywanie, analiza i czyszczenie datasetu z unikalnych wartości i duplikatów.
    """
    training_dataset = pd.read_csv("data/IMDB_Dataset.csv")
    print("==BEFORE CLEANING==")
    print("Number of duplicates: ", training_dataset.duplicated().sum())
    print("Number of null values: ", training_dataset.isnull().sum())
    print("Number of rows: ", training_dataset.shape[0])
    print("Number of columns: ", training_dataset.shape[1])

    print("\n==REVIEW ANALYSIS==")
    print("Type of reviews: ", training_dataset["review"].apply(type).value_counts())
    print("Minimum number of chars: ", training_dataset["review"].str.len().min())
    print("Maximum number of chars: ", training_dataset["review"].str.len().max())
    print("Mean of number of chars", training_dataset["review"].str.len().mean())
    print("Missing values: ", training_dataset["review"].isna().sum())
    print("Unique values: ", training_dataset["review"].nunique())

    print("\n==SENTIMENT ANALYSIS==")
    print("Type of sentiment: ", training_dataset["sentiment"].apply(type).value_counts())
    print("Unique values: ", training_dataset["sentiment"].nunique())
    print("Missing values: ", training_dataset["sentiment"].isna().sum())
    print("Balance of classes: ", training_dataset["sentiment"].value_counts())
    print("Balance of classes (%)", training_dataset["sentiment"].value_counts(normalize=True)*100, "%")

    clean_training = training_dataset.drop_duplicates(subset=["review"]).reset_index(drop=True)
    clean_training["review"] = clean_training["review"].apply(preprocess_base)
    clean_training = clean_training.drop_duplicates(subset=["review"]).reset_index(drop=True)
    clean_training = clean_training[clean_training["review"].str.strip()!= ""]
    clean_training["sentiment"] = clean_training["sentiment"].map({
        "negative":0, 
        "positive":1
    })
    
    print("\n==AFTER CLEANING==")
    print("Number of duplicates: ", clean_training.duplicated().sum())
    print("Number of null values: ", clean_training.isnull().sum())
    print("Number of rows: ", clean_training.shape[0])
    print("Number of columns: ", clean_training.shape[1])
    print(f"Number of kept rows:  {(clean_training.shape[0]/training_dataset.shape[0])*100:.2f}")

    print("\n==REVIEW ANALYSIS==")
    print("Minimum number of chars: ", clean_training["review"].str.len().min())
    print("Maximum number of chars: ", clean_training["review"].str.len().max())
    print("Mean of number of chars", clean_training["review"].str.len().mean())
    print("Missing values: ", clean_training["review"].isna().sum())
    print("Unique values: ", clean_training["review"].nunique())

    print("\n==SENTIMENT ANALYSIS==")
    print("Unique values: ", clean_training["sentiment"].nunique())
    print("Missing values: ", clean_training["sentiment"].isna().sum())
    print("Balance of classes: ", clean_training["sentiment"].value_counts())
    print("Balance of classes (%)", clean_training["sentiment"].value_counts(normalize=True)*100, "%")

    return clean_training