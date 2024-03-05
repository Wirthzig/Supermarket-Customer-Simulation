"""
Description: Main files running the data generation
Authors: Linus Jones, Alex Wirths, Robin Steinkühler
Date: 12.03.2023
"""

from Purchases import Purchases

if __name__ == "__main__":
    # initialize and run generator
    number_of_purchases = 100
    generator = Purchases(number_of_purchases)
    generator.run()

    # export purchase file and fraud value file
    # NOTE: YOU MIGHT NEED TO CHANGE THE FILE NAMES AND LOCATIONS
    path_purchases = "Generated_Datasets/Ah_purchases_Train.csv"
    path_fraud_value = "Generated_Datasets/Fraud_monetary_Train.csv"
    path_customer_types = "Generated_Datasets/Customer_records_Train.csv"
    generator.export_files(path_purchases, path_fraud_value, path_customer_types)
