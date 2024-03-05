"""
Description: Imports the product files
Authors: Linus Jones, Alex Wirths, Robin Steinkühler
Date: 07.03.2023
"""

import pandas as pd
import numpy as np
import os
import json

ignore = ['id', 'category', 'sub category', 'product', 'unit', 'weight', 'calories', 'ingredients']


def get_attr_means_category():
    # get information
    df = get_product_catalog()
    categories = df['category'].unique().tolist()
    cols = list(df.columns)
    attributes = []
    for var in cols:
        if var not in ignore:
            attributes.append(var)
    means = {}

    # calculate means
    for i in categories:
        means[i] = {}
        for j in attributes:
            att = df.groupby('category')[j].mean()
            if not np.isnan(att[i]):
                means[i][j] = att[i]

    return means


def get_attr_mean_sub_category():
    # get information
    df = get_product_catalog()
    sub_categories = df['sub category'].unique().tolist()
    cols = list(df.columns)
    attributes = []
    for var in cols:
        if var not in ignore:
            attributes.append(var)
    means = {}

    # calculate means
    for i in sub_categories:
        means[i] = {}
        for j in attributes:
            att = df.groupby('sub category')[j].mean()
            if not np.isnan(att[i]):
                means[i][j] = att[i]

    return means


def get_product_catalog(path="Data/product_data"):
    """Creates a pandas dataframe from JSON files in a folder hierarchy.
    :param path: The path to the top-level folder.
    :return: A pandas dataframe with observations from the JSON files.
    """
    data, all_keys = [], set()

    # Iterate over each category folder
    for category in os.listdir(path):
        if category == ".DS_Store":
            continue
        category_path = os.path.join(path, category)

        # Iterate over each subcategory folder
        for subcategory in os.listdir(category_path):
            if subcategory == ".DS_Store":
                continue
            subcategory_path = os.path.join(category_path, subcategory)

            # Iterate over each JSON file in the subcategory
            for filename in os.listdir(subcategory_path):
                if filename == ".DS_Store":
                    continue
                elif filename.endswith(".json"):
                    file_path = os.path.join(subcategory_path, filename)

                    # Load the JSON data from the file
                    with open(file_path, "r") as f:
                        json_data = json.load(f)

                    # Add any new keys to the set of all keys
                    for key in json_data:
                        if key not in all_keys:
                            all_keys.add(key)

                    # Add a row to the data list for this file
                    row = {"category": category, "subcategory": subcategory}
                    # Fill in values for existing keys or set them to NaN
                    for key in all_keys:
                        if key in json_data:
                            row[key] = json_data[key]
                        else:
                            row[key] = float('nan')

                    data.append(row)

    # Create a pandas dataframe from the data list
    order = ['id', 'category', 'sub category', 'product', 'price', 'unit', 'weight', 'calories', 'ingredients',
             "lifestyle", "sustainability", "meat", "vegetarian", "vegan"]
    df = pd.DataFrame(data)
    df["pets"] = np.nan
    df["children"] = np.nan

    # Fill ranges for sustainability
    for index, row in df.iterrows():
        if row['category'] == "Pets":
            df.at[index, 'pets'] = 1
        else:
            df.at[index, 'pets'] = 0

        if row['category'] == "Baby and child" or row['category'] == "Sweets, biscuits, crisps and chocolate":
            df.at[index, 'children'] = 1
        else:
            df.at[index, 'children'] = 0
    order.append("pets")
    order.append("children")

    for key in all_keys:
        if key not in order:
            order.append(key)

    # load the json file into a dictionary
    with open('Data/Recipes.json', 'r') as f:
        json_dict = json.load(f)
    # iterate through each key in the dictionary and add a new column to the pandas dataframe
    for key in json_dict.keys():
        df[key] = 0
    # iterate through each row in the pandas dataframe and set the value of each new column
    # to 1 if the item is in the corresponding value list in the dictionary, and 0 otherwise
    for index, row in df.iterrows():
        product_name = row['product']
        for key, value_list in json_dict.items():
            if product_name in value_list:
                df.loc[index, key] = 1
    df = df.astype({'price': float})
    return df[order]