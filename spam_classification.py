# Import LIBs
import joblib
import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Load data
def load_data(path):
    """
    This function loads the dataset
    """
    df = pd.read_csv(path)
    print("Data loaded")
    return df

# Data information
def data_info(df):
    """
    This function prints information about the dataset
    """
    print("Data inforamtion:")
    df.info()

# Text preprocessing using spacy
def spacy_text_preprocessing(text):
    """
    This function preforms preprocessing on text:
        - Lowercasing
        - Tokenization
        - Lemmatization
        - Stopword and punctuation removal
    """
    doc = nlp(text.lower())
    lemma_text = [token.lemma_ for token in doc
                  if token.is_alpha and not token.is_stop]
    return " ".join(lemma_text)

# Apply preprocessing to the whole DataFrame
def preprocess_dataframe(df):
    """
    This function adds a new column 'cleaned_text' with preprocessed text
    """
    print("Preprocessing text data...")
    df["cleaned_text"] = df["Message"].apply(spacy_text_preprocessing)
    print("Text preprocessing completed.")
    return df

# Data preprocessing
def data_preprocessing(df):
    """
    This function split data into text and labels
    """
    return df["cleaned_text"], df["Category"]

# Split the data into train & test
def split_data(text, labels, test_size = 0.2, random_state = 42):
    """
    This function splits the data into train and test sets"""
    return train_test_split(text, labels, test_size = test_size, stratify = labels, random_state = random_state)

# Splited data shape
def splited_data_shape(xtrain, ytrain, xtest, ytest):
    """
    This function returns the shape of train and test sets
    """
    print(f"Train text shape : {xtrain.shape}")
    print(f"Test text shape : {xtest.shape}")
    print(f"Test labels shape : {ytrain.shape}")
    print(f"Test labels shape : {ytest.shape}")

# Pipeline
def build_pipeline(vectorizer, model):
    """
    This function builds the pipeline
    """
    return Pipeline([
        ("vectorizer", vectorizer),
        ("clf", model)
    ])

# Evaluate model
def evaluate_model(model_name, pipeline, xtrain, ytrain, xtest, ytest):
    """
    This function fits the pipeline to train data, test the 
    model on test data, and returns the pipeline and model predictions
    """
    print(f"Evaluating {model_name}")
    pipeline.fit(xtrain, ytrain)
    pipe_pred = pipeline.predict(xtest)
    print("Confusion Matrix:\n", confusion_matrix(ytest, pipe_pred))
    print("Classification Report:\n", classification_report(ytest, pipe_pred))
    return pipeline

# Predict New Message
def predict_message(model):
    new_msg = input("Please enter the message you want to predict:\n")
    prediction = model.predict([new_msg])
    print(f"Model Prediction : {prediction}")



    

def main():
    # Load Data
    df = load_data("F:\Route_AI_Diploma\Eng-Mahmoud_yahia\github_proj\Text_classisication_ML\SPAM text message 20170820 - Data.csv")
    data_info(df)
    df = preprocess_dataframe(df)
    text, labels = data_preprocessing(df)
    print(f"Categories : {labels.value_counts()}")
    xtrain, xtest, ytrain, ytest = split_data(text, labels)
    splited_data_shape(xtrain, ytrain, xtest, ytest)
    
    # Models
    models = {
        "MultinomialNB": MultinomialNB(),
        "ComplementNB": ComplementNB(),
        "LinearSVC": LinearSVC() 
    }

    # Train & Evaluate
    trained_pipelines = {}
    for name, model in models.items():
        pipeline = build_pipeline(TfidfVectorizer(stop_words="english", preprocessor=spacy_text_preprocessing), model)
        trained_pipeline = evaluate_model(name, pipeline, xtrain, ytrain, xtest, ytest)
        trained_pipelines[name] = trained_pipeline
    print("Finished Training")

    # Save the best model
    model_name = input("please type which model to save:\n1-MultinomialNB\n2-ComplementNB\n3-LinearSVC:\n")
    joblib.dump(trained_pipelines[model_name], "spam_classifier.pkl")
    print("Model saved as 'spam_classifier.pkl'")

    # Predict a new sample
    predict_message(trained_pipelines["LinearSVC"])

if __name__ == "__main__":
    main()