"""
Description: Testing the generated dataset with multiple models
Authors: Linus Jones, Alex Wirths, Robin Steinkühler
Date: 10.04.2023
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import random
import lstm3


def model_random(test):
    """ Random model for baseline performance - marks 10% as fraud """
    classifications = [0] * (test.iloc[-1, 1] + 1)
    num_ones = int(0.1 * len(classifications))
    indices = random.sample(range(len(classifications)), num_ones)
    for index in indices:
        classifications[index] = 1
    return classifications


def model_basic_time_sum(train, test):
    """ Looks at the lowest and highest 5% quantiles for the total time (per purchase)"""
    list_1 = train.groupby('purchase id')['time'].sum()
    list_2 = test.groupby('purchase id')['time'].sum()

    quantile_5 = list_1.quantile(0.05)
    quantile_95 = list_1.quantile(0.95)

    classifications = []
    for value in list_2:
        if value < quantile_5:
            classifications.append(1)
        elif value > quantile_95:
            classifications.append(1)
        else:
            classifications.append(0)

    return classifications


def model_basic_price_per_item(test, products):
    """ Model that looks at the highest and lowest 5% quantiles"""
    test["price"] = 0
    for index, row in test.iterrows():
        product_id = row["product id"]
        product_row = products.loc[products['id'] == product_id]
        test.loc[index, "price"] = float(product_row["price"])

    grouped_data = test.groupby('purchase id').agg({'time': 'mean', 'price': 'mean'})
    grouped_data = pd.DataFrame(grouped_data)
    list_1 = grouped_data["price"]

    quantile_5 = list_1.quantile(0.05)
    quantile_95 = list_1.quantile(0.95)

    classifications = []
    for value in list_1:
        if value < quantile_5:
            classifications.append(1)
        elif value > quantile_95:
            classifications.append(1)
        else:
            classifications.append(0)

    return classifications


def neural_network(train, test):
    """ LSTM neural network """
    return lstm3.main(train, test)


def performance(pred, true, name):
    """ Prints the accuracy, auc, and detected fraud value of each model. """
    value_caught, true_label = 0, []

    # calculate values
    for i in range(len(pred)):
        if pred[i] == 1:
            value_caught += true[i]

        if true[i] == 0.0:
            true_label.append(0)
        else:
            true_label.append(1)

    # print results
    print("Model: ", name)
    print("Accuracy: ", accuracy_score(true_label, pred))
    print("AUC: ", roc_auc_score(true_label, pred))
    print("Fraud value detected: ", value_caught, " of ", np.sum(true))
    print("-----------------------------------------------")


if __name__ == "__main__":
    # read datasets
    # NOTE: YOU MIGHT NEED TO CHANGE THE FILE NAME (AND LOCATION)
    X_train = pd.read_csv("Generated_Datasets/AH_purchases_Train.csv")
    X_test = pd.read_csv("Generated_Datasets/Ah_purchases_Test.csv")
    y_test = pd.read_csv("Generated_Datasets/Fraud_monetary_Test.csv")
    y_test = list(y_test["amount"])
    product_data = pd.read_csv("Generated_Datasets/Product_Catalog.csv", index_col=0)

    # run models
    m1 = model_random(X_test)
    m2 = model_basic_price_per_item(X_test, product_data)
    m3 = model_basic_time_sum(X_train, X_test)
    m4 = neural_network(X_train, X_test)

    # performance
    performance(m1, y_test, "random")
    performance(m2, y_test, "price")
    performance(m3, y_test, "time")
    performance(m4, y_test, "neural network")
