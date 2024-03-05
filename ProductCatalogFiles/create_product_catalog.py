
import pandas as pd
import numpy as np
import random
import os
import json


def get_basis(path):
    """ Modify product data """
    df = pd.read_csv(path)

    # Remove the / in the product names
    df["product"] = df["product"].str.replace("/", " en ")
    # Remove duplicate products
    df['names'] = df['product'].str.lower()
    df = df.drop_duplicates(subset=['names'])
    # Drop unnecessary columns
    data = df[['id', 'category', 'sub category', 'product', 'price', 'organic', 'vegetarian', 'vegan', 'gluten free', 'lactose free', 'unit', 'weight', 'calories', 'ingredients']]
    # Add new columns
    data = pd.DataFrame(data)
    data["meat"] = np.nan
    data["vegetarian"] = np.nan
    data["vegan"] = np.nan
    data["lifestyle"] = np.nan
    data["sustainability"] = np.nan
    data["pets"] = np.nan
    data["children"] = np.nan
    data["fraud probability"] = np.nan
    data["price sensitivity"] = np.nan

    # create price sensitivity
    def normalize(x):
        x_min = x.min()
        x_max = x.max()
        result = (2 * (x - x_min) / (x_max - x_min)) - 1
        return result

    # apply the normalization function to each group separately
    data["price sensitivity"] = data.groupby('category', group_keys=True)['price'].apply(normalize).tolist()

    # create fraud probability
    def price_to_prob(x):
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min)

    # apply function to each group
    data["fraud probability"] = data.groupby('category')['price'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Drop unnecessary categories and sub categories
    no_categories = ["Cooking, dining, leisure", "Household", "Drugstore"]
    no_sub_categories = ["Baby and child care", "Diapers and changing"]
    data = data[~data['category'].isin(no_categories)]
    data = data[~data['sub category'].isin(no_sub_categories)]

    # Fill ranges for sustainability
    for index, row in data.iterrows():
        if row['organic'] == 1:
            data.at[index, 'sustainability'] = 1
        else:
            data.at[index, 'sustainability'] = 0

    # Fill ranges for meant, vegetarian, vegan
    # Define category ranges
    vegan_sub_categories = ["Alcohol-free beer and beer low in alcohol", "Bake-off bread", "Beer", "Couscous, bulgur, quinoa, groats", "Frozen vegetables", "Fruit", "Potatoes", "Red wine", "Rosé", "Salads", "Soft drinks", "Special beers", "Spices", "Sports drinks", "Tea", "Vegetables", "White wine"]
    vegetarian_sub_categories = ["Baby food", "Biscuits", "Breakfast cereals", "Butter", "Cheese", "Coffee", "Crackers, rice cakes", "Easter treats", "Fruit bars", "Fruit biscuits & milk biscuits", "Ice cream", "Milk", "Nut bars", "Pasta, rice, noodles", "Pastries", "Preserves", "Protein bars", "Protein shakes", "Rodents", "Sweet spreads", "Sweets", "Yoghurt & cottage cheese"]
    meat_sub_categories = ["Cats", "Chicken", "Dogs", "Fish", "Meat", "Processed meats", "Snacks", "Soups"]
    # TODO DETERMINE TOGETHER HOW TO CLASSIFY THEM
    diet_undefined_subcategories = ["AH Fresh from home", "Fresh ready-to-eat meals", "Frozen snacks", "Meal packs, mixes"]

    for index, row in data.iterrows():
        if row['vegan'] == 1:
            data.at[index, 'vegan'] = 1
            data.at[index, 'vegetarian'] = 1
            data.at[index, 'meat'] = -1
        elif row['vegetarian'] == 1:
            data.at[index, 'vegan'] = 0
            data.at[index, 'vegetarian'] = 1
            data.at[index, 'meat'] = -1
        elif row["sub category"] in vegan_sub_categories:
            data.at[index, 'vegan'] = 1
            data.at[index, 'vegetarian'] = 1
            data.at[index, 'meat'] = -1
        elif row["sub category"] in vegetarian_sub_categories:
            data.at[index, 'vegan'] = 0
            data.at[index, 'vegetarian'] = 1
            data.at[index, 'meat'] = -1
        elif row["sub category"] in meat_sub_categories:
            data.at[index, 'vegan'] = -1
            data.at[index, 'vegetarian'] = -1
            data.at[index, 'meat'] = 1
        elif row["sub category"] in diet_undefined_subcategories:
            data.at[index, 'vegan'] = 0
            data.at[index, 'vegetarian'] = 0
            data.at[index, 'meat'] = 0

    # Remove obsolete columns
    data = data[['id', 'category', 'sub category', 'product', 'price', 'unit', 'weight', 'calories', 'ingredients', 'lifestyle', 'vegan', 'vegetarian', 'meat', 'sustainability', 'fraud probability', 'price sensitivity']]
    # Define category ranges
    unhealthy = ["Special beers", "Beer", "Rosé", "White wine", "Red wine", "Frozen snacks", "Ice cream", "Fruit biscuits & milk biscuits", "Sweets", "Cookies", "Easter treats", "Crackers, rice cakes", "Sweet spreads"]
    neutral = ["Alcohol-free beer and beer low in alcohol", "Soft drinks", "Rodents", "Cats", "Dogs", "Spices"]
    healthy = ["Tea", "Coffee", "Frozen vegetables", "Protein shakes", "Protein bars", "Sports drinks", "Couscous, bulgur, quinoa, groats", "Salads", "Fruit", "Vegetables", "Baby food"]
    # TODO DETERMINE TOGETHER HOW TO CLASSIFY THEM
    unclassified = ["Preserves",
                    "Soups", "Meal packs, mixes", "Pasta, rice, noodles", "Nut bars", "Fruit bars",
                    "Breakfast cereals", "Bake-off bread", "Pastries", "Butter", "Yoghurt & cottage cheese", "Processed meats", "Cheese",
                    "Fish", "Chicken", "Meat", "Fresh ready-to-eat meals", "AH Fresh from home", "Potatoes", "Snacks", "Biscuits", "Milk"]

    for index, row in data.iterrows():
        if row["sub category"] in unhealthy:
            data.at[index, 'lifestyle'] = -1
        elif row["sub category"] in neutral:
            data.at[index, 'lifestyle'] = 0
        elif row["sub category"] in healthy:
            data.at[index, 'lifestyle'] = 1
        elif row["sub category"] in unclassified:
            data.at[index, 'lifestyle'] = round(random.uniform(-1, 1), 1)

    # TODO: still missing pets and children

    return data


def reset_directory(path="ProductCatalogFiles/AH_product_dataset_EN.csv", setup=False):
    """ Creates the files system """
    # WARNING MESSAGE
    if not setup:
        response = input("WARNING: This function will reset the directory. All attributes that were manually added in the text files will be deleted. Products that were added manually will still exist. \n Proceed? [y/n]")
    else:
        response = "y"

    # create product file system
    if response.lower() == "y":
        df = get_basis(path)

        if not os.path.exists("../Data/product_data"):
            os.mkdir("../Data/product_data")

        for index, row in df.iterrows():
            # Extract information from dataset
            category = row["category"]
            sub_category = row["sub category"]
            name = row["product"]

            # Create folders
            if not os.path.exists("product_data/" + str(category)):
                os.mkdir("product_data/" + str(category))
            if not os.path.exists("product_data/" + str(category) + "/" + str(sub_category)):
                os.mkdir("product_data/" + str(category) + "/" + str(sub_category))

            # Create path
            temp_path = "product_data/" + str(category) + "/" + str(sub_category) + "/" + str(name) + ".json"

            # Write json files
            with open(temp_path, "w") as f:
                json.dump(dict(row), f, indent=4)

        print("finished!")

    elif response.lower() != "n":
        print("Invalid input. Please enter 'y' or 'n'.")
        reset_directory()


if __name__ == "__main__":
    reset_directory(setup=True)
