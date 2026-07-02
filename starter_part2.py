import numpy as np
import pandas as pd
import gensim.downloader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Pre-trained embedding model (do not modify)
model = gensim.downloader.load("word2vec-google-news-300")
EMBEDDING_DIM = 300


def document_vector(text):
    """
    Convert text into a single vector by averaging word embeddings.

    
    Tokenizes the text based on whitespace.
    Ignores out-of-vocabulary (OOV) words.
    If all words are OOV, returns a zero vector with shape (300,).

    Args:
        text (str): input review text

    Returns:
        np.ndarray: document vector with shape (300,)
    """

    words = text.split()
    vectors = []
    for i in range(len(words)):
        w = words[i]
        if w in model:
            vectors.append(model[w])
    
    if len(vectors) == 0:
        return np.zeros(EMBEDDING_DIM)
    
    # calculate average
    total = np.zeros(EMBEDDING_DIM)
    for v in vectors:
        total = total +v
    result = total /len(vectors)

    return result
    


def prepare_classification_data(file_path):
    """
    Loads the IMDB dataset and convert it into feature and label arrays.

    Expected CSV format:
        review,sentiment

    Label mapping:
        positive -> 1
        negative -> 0

    Args:
        file_path (str): path to the CSV file

    Returns:
        X (np.ndarray): feature matrix with shape (N, 300)
        y (np.ndarray): label vector with shape (N,)
    """
  
    # read csv and prepare data
    df = pd.read_csv(file_path)
    
    # Convert labels to 0 and 1

    labels = []
    for i in range(len(df)):
        if df['sentiment'].iloc[i] =='positive':
            labels.append(1)
        else:
            labels.append(0)

    # calculate vector for each review
    X_list = []
    for i in range(len(df)):
        review_text = df['review'].iloc[i]
        vec = document_vector(review_text)
        X_list.append(vec)

    X = np.array(X_list)
    y = np.array(labels)

    return X, y


def train_classifier(X_train, y_train,
                     hidden_dim=64,
                     learning_rate=0.001,
                     epochs=10,
                     batch_size=32):
    """
    Trains a simple neural network classifier using document vectors.

    
    Uses a deep learning framework such as PyTorch.
    The model has:
        * an input layer matching the embedding dimension,
        * at least one hidden layer,
        * an output layer for binary classification.
    The function returns the trained model.

    Args:
        X_train (np.ndarray): training feature matrix
        y_train (np.ndarray): training labels

    Returns:
        torch.nn.Module: trained neural network classifier
    """

    mahmutmodel = nn.Sequential(
        nn.Linear(300, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid()
    )

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.BCELoss() # LOSS funcktion
    optim = torch.optim.Adam(mahmutmodel.parameters(), lr=learning_rate)

    for ep in range(epochs):
        for bx, by in loader:
            optim.zero_grad()
            out = mahmutmodel(bx)
            loss = loss_fn(out, by)
            loss.backward()
            optim.step()
        # print("epoch", ep, "loss:", loss.item())
    
    return mahmutmodel


def evaluate_classifier(clf, X_test, y_test):
    """
    Evaluates the trained classifier.

    
    Predicts labels for X_test.
    Computes accuracy, precision, recall, and F1-score.

    Args:
        clf (torch.nn.Module): trained classifier
        X_test (np.ndarray): test feature matrix
        y_test (np.ndarray): test labels

    Returns:
        dict: {
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1": float
        }
    """
    # evaluate on test data
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        out = clf(X_tensor)
    
    preds = torch.round(out).numpy()
    #metricss
    acc = accuracy_score(y_test,preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test,preds)
    
    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

    return results
