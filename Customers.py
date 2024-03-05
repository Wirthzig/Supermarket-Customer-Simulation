"""
Description: Class Customer
Authors: Linus Jones, Alex Wirths, Robin Steinkühler
Date: 07.03.2023
"""
import random

import numpy as np
import math


class Customer:
    """
    Class defines one customer type

    Attributes:
        id (int): the customer type id
        name: the customer type name
        fraud_prob (float): the fraud probability of the customer type
        time_factor (float): A factor indicating how fast/slow a customer type is
        amount_factor (float): A factor indicating how many products a customer type buys
        recipe (String): The recipe of the customer that is chosen
        attributes (dict): The customer type attributes such as price sensitivity, lifestyle, vegan, and more
    """

    def __init__(self, cust_id, args):
        """ Stores attributes and data about customer type
        :param cust_id: the id of the customer type
        :param args: a dictionary containing all the customer type values
        """
        # general customer type information
        self.id = cust_id
        self.name = args["name"]
        self.fraud_prob = args["fraud_probability"]

        # attributes concerning purchases itself
        self.time_factor = self.create_factor(self.value_normal_dist(args["time"]["mean"], args["time"]["std"]))
        self.amount_factor = self.create_factor(self.value_normal_dist(args["amount"]["mean"], args["amount"]["std"]))
        self.recipe = None
        self.choose_recipe(args["recipes"])
        self.attributes = None
        self.set_attributes(args["attributes"])

    def set_attributes(self, vals):
        """ Initialize, sets and store attributes
        :param vals: the attributes and their mean and std values
        """
        # assign value from mean and std
        args = vals.copy()
        for key in args.keys():
            args[key] = self.value_normal_dist(args[key]["mean"], args[key]["std"])
        self.attributes = args

    def choose_recipe(self, args):
        """ Picks one recipe from the available ones
        :param args: the recipes and their corresponding probabilities to be chosen
        """
        options, probs = [None], [0.2]
        for key in args.keys():
            options.append(key)
            probs.append(args[key])

        self.recipe = random.choices(options, weights=probs)[0]

    @staticmethod
    def create_factor(value):
        """ Returns a factor that is derived from a given value
        :param value: The value (float) that is transformed
        :return: the factor (float) that is derived
        """
        if value < 0:
            return 1 / math.exp(abs(value))
        elif value > 0:
            return math.exp(value)
        else:
            return 0

    @staticmethod
    def value_normal_dist(mean, std):
        """ Returns a random value from a normal distribution  while enforcing the [-1,1]
        :param mean: the mean value
        :param std: the standard deviation
        :return: the random value from the distribution
        """
        value = np.random.normal(mean, std)
        # enforcing the interval
        if value < -1:
            return -1
        elif value > 1:
            return 1
        else:
            return value
