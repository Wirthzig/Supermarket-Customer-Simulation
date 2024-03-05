"""
Description: Class Purchases
Authors: Linus Jones, Alex Wirths, Robin Steinkühler
Date: 07.03.2023
"""

import pandas as pd
import numpy as np
import random
import os
import json
from Customers import Customer
import product_catalog
import store_layout
from tqdm import tqdm


class Purchases:
    """
    The class contains methods to read and create all customer types, create the purchase dataset, and introduce fraud.

    Attributes:
        num_purchases (int): The number of purchases (visits to the store)
        purchases (dataframe): The purchase information given by the purchase id, product id, and the time
        record_customers (list): Contains the index of the customers that made each of purchases
        fraud_value_per_purchase (list): Contains the fraud in monetary terms per purchase
        customers (list): Contains all the Customer class objects (the customer types )
        customers_fraud_prob (list): Contains the fraud probability of each customer type
        shop_layout_ids (dict): Contains the id of each department
        shop_layout_ids2 (dict): Contains the id of each department (different representation)
        shop_layout_times (list): The adjacency matrix containing the time needed to move between departments/categories
        products (dataframe): Contains all the product information
        attr_means_categories (dict): Contain the mean attribute values for each category
        attr_means_sub_categories (dict): Contains the mean attribute values for each sub-category
        recipes (dict): Contains all of the recipes (name and list of ingredients)
    """

    def __init__(self, num_purchases: int):
        """ Initialises the class by getting all the needed information and creating storage space
        :param num_purchases: the number of purchases (visits to the store) that are desired
        """
        # Stores DataFrame containing all simulated purchases and records the customer type used for all purchases
        self.num_purchases = num_purchases
        self.purchases = pd.DataFrame(columns=["purchase id", "product id", "time"])
        self.record_customers = []
        self.fraud_value_per_purchase = [0] * num_purchases

        # Stores customer objects and their probabilities of fraud
        self.customers = []
        self.customers_fraud_prob = []

        # Department ids and shop layout (adjacency matrix with distances)
        self.shop_layout_ids = store_layout.adj_ids
        self.shop_layout_ids2 = store_layout.adj_ids2
        self.shop_layout_times = store_layout.get_distance_matrix()

        # Product dataset
        self.products = product_catalog.get_product_catalog()
        self.attr_means_categories = product_catalog.get_attr_means_category()
        self.attr_means_sub_categories = product_catalog.get_attr_mean_sub_category()
        with open("Data/Recipes.json") as f:
            self.recipes = json.load(f)

    def run(self):
        """ Runs all the methods for creating the fraudulent product dataset """
        self.create_customer_types("Data/CustomerTypes")
        print("** Setup completed **")
        print("Purchase Simulation:")
        self.purchase_simulation()
        print("Fraud introduction: ")
        self.introduce_fraud()
        print("** Dataset generated **")

    def create_customer_types(self, path):
        """ Creates n different customers (customer types) and stores them in the class attributes
        :param path: path to where the customer type files are stored
        """
        counter = 0
        # Loop over all files in the directory and open all json files
        for filename in os.listdir(path):
            if filename.endswith('.json'):
                with open(os.path.join(path, filename), 'r') as f:
                    # get specified attributes and create customer type
                    arg_vals = json.load(f)

                    for i in range(arg_vals["n"]):  # create n times
                        cust = Customer(cust_id=counter, args=arg_vals)
                        # store relevant information
                        self.customers.append(cust)
                        self.customers_fraud_prob.append(cust.fraud_prob)
                        counter += 1

    def purchase_simulation(self):
        """ Combines methods to generate the complete purchase dataset with n entries """
        options_indices = list(range(len(self.customers)))  # options of customers to choose from

        for i in tqdm(range(self.num_purchases)):
            # pick random customer
            customer_index = random.choice(options_indices)
            # simulate purchase for the chosen customer
            self.create_purchase(customer=self.customers[customer_index], purchase_id=i)
            self.record_customers.append(customer_index)

    def create_purchase(self, customer, purchase_id):
        """ Simulates one purchase (one visit to the store) by a given customer
        :param customer: a Customer class object (to get the customer attributes)
        :param purchase_id: the id of the current visit to the store
        """
        # adding information about recipes
        if customer.recipe is not None:
            aux = self.products[self.products["product"].isin(self.recipes[customer.recipe])]
            aux_cat = aux["category"].tolist()
            aux_sub_cat = aux["sub category"].tolist()
            aux_cat_id = [self.shop_layout_ids[i] for i in aux_cat]
            aux_sub_cat_id = [self.shop_layout_ids2[i] for i in aux_sub_cat]
            num_products_cut = [2 for _ in range(18)]
            num_products_sub_cut = [2 for _ in range(51)]

            for i in aux_cat_id:
                if num_products_cut[i] > 1:
                    num_products_cut[i] -= 1

            for i in aux_sub_cat_id:
                if num_products_sub_cut[i] > 1:
                    num_products_sub_cut[i] -= 1

        else:
            num_products_cut = [1 for _ in range(18)]
            num_products_sub_cut = [1 for _ in range(51)]

        # number of items bought per category (starts with one)
        category = 'Potatoes, vegetables, fruit'
        exit_prob = 0.025
        num_items = 15 * customer.amount_factor
        counter_products = 0

        # the customer goes shopping...
        while True:
            # Step 1. Choose category
            # Calculates the distance/similarity between the customer and the categories
            cat, distances = self.distance_category(customer=customer)
            distances = [i / j for i, j in zip(distances, num_products_cut)]  # adjust by number of products bought

            # Retrieves the times to get from the current department to all other departments
            times = self.shop_layout_times[self.shop_layout_ids[category]]

            # Converts the distances and times to combined probabilities and picks based on that the next category
            category_probabilities = self.convert_to_probabilities(distances=distances, times=times,
                                                                   exit_prob=exit_prob)
            prev_category = category
            category = random.choices(cat, weights=category_probabilities)[0]

            # If the customer decides to exit the store, shopping is stopped
            if category == 'Exit':
                if counter_products < 2:  # in case no products have been bought so far
                    category = 'Potatoes, vegetables, fruit'
                    continue
                else:
                    break
            else:
                # Inc the number of products bought in the category
                num_products_cut[self.shop_layout_ids[category]] += 1
                counter_products += 1
                # adjust exit probability
                if sum(num_products_cut) > 18 + num_items:
                    exit_prob += 0.1

            # Step 2. Choose sub-category
            # Calculates the distance between the customer and the sub-category
            cat, distances = self.distance_sub_category(category=category, customer=customer)
            distances = [i / j for i, j in zip(distances, num_products_sub_cut)]
            # Converts the attribute distances to probabilities (Time is neglected as we are in the same category)
            sub_category_probabilities = self.convert_to_probabilities(distances=distances)
            sub_category = random.choices(cat, weights=sub_category_probabilities)[0]  # pick random sub-category
            # Inc the number of products bought in the sub category
            num_products_sub_cut[self.shop_layout_ids2[sub_category]] += 1

            # Step 3. Choose product
            # get products, calculate similarity/distance and choose one depending on probabilities
            product_options = self.products[self.products["sub category"] == sub_category]
            product_distance = self.distance_product(customer, product_options)
            product_probs = self.convert_to_probabilities(distances=product_distance)
            product_index = random.choices(list(range(len(product_options))), weights=product_probs)[0]
            product = product_options.iloc[product_index]

            # Determine the time it took to scan the item
            time_to_cat = self.shop_layout_times[self.shop_layout_ids[prev_category]][self.shop_layout_ids[category]]
            time = time_to_cat * customer.time_factor

            # store purchases product
            new_row = pd.Series({"purchase id": int(purchase_id), "product id": int(product.iloc[0]),
                                 "time": round(time)})
            self.purchases = pd.concat([self.purchases, new_row.to_frame().T], ignore_index=True)

    @staticmethod
    def attribute_distance(x, y, sensitivity=10):
        """ Calculates the distance between two attributes
        :param x: attribute of the customer
        :param y: attribute of the product
        :param sensitivity: sensitivity of the function
        :return: the distance between the two attributes
        """
        k = abs(x - y)
        if k == 2 or k == -2:
            distance = sensitivity * k
        else:
            distance = sensitivity * k / ((k - 2) * (k + 2))
        return abs(distance)

    def distance_category(self, customer):
        """Calculates the distance of a customers attributes to a category. The category attributes represent the
        mean over all the products they contain.
        :param customer: a Customer class object (to get the customer attributes)
        :return: a list with the categories and a list with the similarities/distances to each of the categories
        """
        distance, cat = [], []

        # loop over categories
        for i in self.attr_means_categories:
            dist = 0
            for key in customer.attributes:
                if key in self.attr_means_categories[i]:
                    x, y = customer.attributes[key], self.attr_means_categories[i][key]
                    # calculate the sum of the squared differences
                    dist += self.attribute_distance(x, y)

            # take the square root of the sum of the squared differences
            distance.append(dist ** (1 / 2))
            cat.append(i)
        cat.append('Exit')

        return cat, distance

    def distance_sub_category(self, customer, category):
        """Calculates the distance of a customers attributes to a sub-category. The sub-categories attributes
        represent the mean over all the products they contain.
        :param customer: a Customer class object (to get the customer attributes)
        :param category: the current category of the customer (to get its sub-categories)
        :return: a list with the sub-categories and a list with the similarities/distances to each of the sub-categories
        """
        distance, cat = [], []
        dic = {'Baby and child': ['Baby food'],
               'Bakery and pastry': ['Bake-off bread', 'Pastries'],
               'Beer and aperitifs': ['Alcohol-free beer and beer low in alcohol', 'Special beers', 'Beer'],
               'Breakfast cereals and spreads': ['Crackers, rice cakes', 'Sweet spreads', 'Breakfast cereals'],
               'Cheese, cold cuts, tapas': ['Snacks', 'Processed meats', 'Cheese'],
               'Dairy, vegetable and eggs': ['Butter', 'Yoghurt & cottage cheese', 'Milk'],
               'Frozen foods': ['Frozen vegetables', 'Frozen snacks', 'Ice cream'],
               'Meat, chicken, fish, vega': ['Fish', 'Chicken', 'Meat'],
               'Pasta, rice and world cuisine': ['Meal packs, mixes', 'Couscous, bulgur, quinoa, groats',
                                                 'Pasta, rice, noodles'],
               'Pets': ['Rodents', 'Cats', 'Dogs'],
               'Potatoes, vegetables, fruit': ['Potatoes', 'Fruit', 'Vegetables'],
               'Salads, pizza, meals': ['Fresh ready-to-eat meals', 'Salads', 'AH Fresh from home'],
               'Snacks': ['Nut bars', 'Fruit bars', 'Fruit biscuits & milk biscuits'],
               'Soft drinks, juices, coffee, tea': ['Soft drinks', 'Tea', 'Coffee'],
               'Soups, sauces, condiments, oils': ['Spices', 'Preserves', 'Soups'],
               'Sports and dietary foods': ['Protein shakes', 'Protein bars', 'Sports drinks'],
               'Sweets, biscuits, crisps and chocolate': ['Sweets', 'Biscuits', 'Easter treats'],
               'Wine and bubbles': ['Rosé', 'White wine', 'Red wine']}

        # loop over all sub-categories within current category and calculate their similarity to the customer
        sub_cats = {key: self.attr_means_sub_categories[key] for key in dic[category]}
        for i in sub_cats:
            dist = 0
            for key in customer.attributes:  # calculate dist for each customer attribute if also present in sub-cat
                if key in self.attr_means_sub_categories[i]:
                    # calculate the sum of the squared differences
                    x, y = customer.attributes[key], self.attr_means_sub_categories[i][key]
                    dist += self.attribute_distance(x, y)

            # take the square root of the sum of the squared differences
            distance.append(dist ** (1 / 2))
            cat.append(i)

        return cat, distance

    def distance_product(self, customer, product_options):
        """Calculates the distance of a customers attributes to products.
        :param customer: a Customer class object (to get the customer attributes)
        :param product_options: dataframe of products that are available
        :return: a list with the similarities/distances to each of the categories
        """
        distance = []
        ignore = ['id', 'category', 'sub category', 'product', 'unit', 'weight', 'calories', 'ingredients']
        attributes = list(self.products.columns)

        # loop over categories
        for index, product in product_options.iterrows():
            dist = 0
            for key in attributes:
                if key not in ignore:
                    if key in customer.attributes:
                        x, y = customer.attributes[key], product.loc[key]
                        # calculate the sum of the squared differences
                        dist += self.attribute_distance(x, y)

            # take the square root of the sum of the squared differences
            distance.append(dist ** (1 / 2))

        return distance

    @staticmethod
    def convert_to_probabilities(distances, times=None, exit_prob=None):
        """ Converts distances and times into probabilities. Higher distances result in lower
        probabilities and vice versa. The sum of all probabilities is 1.
        :param distances: list containing the distance/similarity between the customer and the categories
        :param times: list containing the time it takes to get from the current department to all other departments
        :param exit_prob: the exit probability of the customer
        :return: list of probabilities to go from the current category to every other category
        """
        probabilities = []

        # Calculates the probabilities based on the distances
        if times is None:
            for i in distances:
                if i == 0:
                    probabilities.append(1)
                else:
                    probabilities.append(1 / i)

            # normalise probabilities (so they add up to one)
            total = sum(probabilities)
            return [x / total for x in probabilities]

        # Calculates the probabilities based on both distances and times and weighs them accordingly
        else:
            for i, j in zip(distances, times):
                if i == 0:
                    val = 1
                else:
                    val = 1 / i
                if j == 0:
                    val2 = 1
                else:
                    val2 = 1 / j
                probabilities.append(val * val2)  # combines probabilities

            # normalise probabilities (so they add up to one)
            total = sum(probabilities)
            probabilities = [x / total for x in probabilities]
            # add exit probability
            probabilities.append(exit_prob)
            return probabilities

    def introduce_fraud(self):
        """ Introduces fraud into the purchase dataset """

        # count the number of products per purchase/visit
        product_counts = [0] * self.num_purchases
        for i in range(len(self.purchases)):
            product_counts[self.purchases.iloc[i, 0]] += 1

        # introduces fraud to purchases
        position, drop_rows = 0, []
        for i in tqdm(range(self.num_purchases)):
            # make decision on whether to introduce fraud or not based on customer type fraud probability
            # check that there are enough products
            if random.random() < self.customers_fraud_prob[self.record_customers[i]] and product_counts[i] > 2:
                # get the purchased products  and their corresponding fraud probabilities
                product_ids = self.purchases.loc[position:position + product_counts[i] - 1, "product id"]
                fraud_probs = []
                for j in product_ids:
                    val = self.products.loc[self.products["id"] == j, "fraud probability"]
                    fraud_probs.append(float(val))

                # pick number of items and the exact items based on probability to make fraudulent
                num_fraud = np.random.poisson(3)
                if num_fraud >= product_counts[i]:
                    num_fraud = product_counts[i] - 1
                fraud_items = random.choices(list(range(len(product_ids))), weights=fraud_probs, k=num_fraud)
                fraud_items = list(np.unique(fraud_items))  # in case there are duplicates

                # apply fraud method: pick fraud type depending on probability
                if random.random() < 0.2:  # banana
                    for item in fraud_items:
                        # get current product information
                        current_item_info = self.products.loc[self.products["id"] == product_ids.iloc[item]]
                        cat = current_item_info.iloc[0, 1]
                        price = current_item_info.iloc[0, 4]

                        # get all cheaper items within the same category and pick one randomly
                        cat_items = self.products.loc[(self.products["category"] == cat)]
                        relevant_items = cat_items.loc[(self.products["price"] < price)]

                        if len(relevant_items) > 0:
                            new_item_index = random.choice(list(range(len(relevant_items))))
                        else:
                            continue

                        # add new item information to purchase dataset and record monetary fraud value
                        self.purchases.iloc[position + item, 1] = relevant_items.iloc[new_item_index, 0]
                        self.purchases.iloc[position + item, 2] += round(abs(np.random.normal(15, 5)))
                        self.fraud_value_per_purchase[i] += price - relevant_items.iloc[new_item_index, 4]

                else:  # pass-around
                    for item in fraud_items:
                        item_position = position + item

                        # increase time for following purchase and mark item as dropped
                        if item_position+1 < len(self.purchases):
                            if self.purchases.iloc[item_position, 0] == self.purchases.iloc[item_position + 1, 0]:
                                self.purchases.iloc[item_position + 1, 2] += self.purchases.iloc[item_position, 2]
                            drop_rows.append(item_position)

                        # record monetary fraud value by getting product price from product dataset
                        item_id = self.purchases.iloc[item_position, 1]
                        self.fraud_value_per_purchase[i] += float(
                            self.products.loc[self.products["id"] == item_id, "price"])

            # increase position in purchase dataframe
            position += product_counts[i]

        # drop stored rows (fraud items in pass-around trick)
        self.purchases.drop(drop_rows, inplace=True, axis=0)

    def export_files(self, name1, name2, name3):
        """ Exports the purchase dataset and whether purchases are fraudulent (in monetary value) as two csv files """
        self.purchases.to_csv(name1)
        df = pd.DataFrame(self.fraud_value_per_purchase)
        df.columns = ["amount"]
        df.to_csv(name2)

        customer_names = []
        for i in self.record_customers:
            customer_names.append(self.customers[i].name)
        pd.DataFrame(customer_names).to_csv(name3)
