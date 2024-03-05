"""
Description: Script generates the AH shop layout
Authors: Linus Jones, Alex Wirths, Robin Steinkühler
Date: 03.03.2023
"""

import numpy as np
import random

# Define the categories
categories = ['Pets', 'Baby and child', 'Frozen foods', 'Sports and dietary foods', 'Soups, sauces, condiments, oils',
              'Pasta, rice and world cuisine', 'Beer and aperitifs', 'Wine and bubbles',
              'Soft drinks, juices, coffee, tea',
              'Snacks', 'Sweets, biscuits, crisps and chocolate', 'Breakfast cereals and spreads', 'Bakery and pastry',
              'Dairy, vegetable and eggs', 'Cheese, cold cuts, tapas', 'Meat, chicken, fish, vega',
              'Salads, pizza, meals',
              'Potatoes, vegetables, fruit']

# Define the adjacency list
adj_list = {
    'Pets': ['Baby and child', 'Salads, pizza, meals', 'Meat, chicken, fish, vega', 'Potatoes, vegetables, fruit'],
    'Baby and child': ['Pets', 'Salads, pizza, meals', 'Meat, chicken, fish, vega', 'Potatoes, vegetables, fruit'],
    'Frozen foods': ['Bakery and pastry', 'Meat, chicken, fish, vega'],
    'Sports and dietary foods': ['Pasta, rice and world cuisine', 'Salads, pizza, meals', 'Potatoes, vegetables, fruit'],
    'Soups, sauces, condiments, oils': ['Breakfast cereals and spreads', 'Salads, pizza, meals', 'Potatoes, vegetables, fruit'],
    'Pasta, rice and world cuisine': ['Sports and dietary foods', 'Salads, pizza, meals', 'Potatoes, vegetables, fruit'],
    'Beer and aperitifs': ['Wine and bubbles', 'Dairy, vegetable and eggs', 'Meat, chicken, fish, vega'],
    'Wine and bubbles': ['Beer and aperitifs', 'Dairy, vegetable and eggs', 'Meat, chicken, fish, vega'],
    'Soft drinks, juices, coffee, tea': ['Bakery and pastry', 'Dairy, vegetable and eggs', 'Cheese, cold cuts, tapas'],
    'Snacks': ['Sweets, biscuits, crisps and chocolate', 'Dairy, vegetable and eggs', 'Meat, chicken, fish, vega'],
    'Sweets, biscuits, crisps and chocolate': ['Snacks', 'Dairy, vegetable and eggs', 'Meat, chicken, fish, vega'],
    'Breakfast cereals and spreads': ['Soups, sauces, condiments, oils', 'Salads, pizza, meals', 'Potatoes, vegetables, fruit'],
    'Bakery and pastry': ['Frozen foods', 'Soft drinks, juices, coffee, tea', 'Cheese, cold cuts, tapas', 'Meat, chicken, fish, vega'],
    'Dairy, vegetable and eggs': ['Beer and aperitifs', 'Wine and bubbles', 'Soft drinks, juices, coffee, tea', 'Snacks', 'Sweets, biscuits, crisps and chocolate', 'Cheese, cold cuts, tapas'],
    'Cheese, cold cuts, tapas': ['Soft drinks, juices, coffee, tea', 'Bakery and pastry', 'Dairy, vegetable and eggs'],
    'Meat, chicken, fish, vega': ['Pets', 'Baby and child', 'Frozen foods', 'Beer and aperitifs', 'Wine and bubbles', 'Snacks', 'Sweets, biscuits, crisps and chocolate', 'Bakery and pastry', 'Potatoes, vegetables, fruit'],
    'Salads, pizza, meals': ['Pets', 'Baby and child', 'Sports and dietary foods', 'Soups, sauces, condiments, oils', 'Pasta, rice and world cuisine', 'Breakfast cereals and spreads'],
    'Potatoes, vegetables, fruit': ['Pets', 'Baby and child', 'Sports and dietary foods', 'Soups, sauces, condiments, oils', 'Pasta, rice and world cuisine', 'Breakfast cereals and spreads', 'Meat, chicken, fish, vega']
}

adj_ids = {
    'Pets': 0,
    'Baby and child': 1,
    'Frozen foods': 2,
    'Sports and dietary foods': 3,
    'Soups, sauces, condiments, oils': 4,
    'Pasta, rice and world cuisine': 5,
    'Beer and aperitifs': 6,
    'Wine and bubbles': 7,
    'Soft drinks, juices, coffee, tea': 8,
    'Snacks': 9,
    'Sweets, biscuits, crisps and chocolate': 10,
    'Breakfast cereals and spreads': 11,
    'Bakery and pastry': 12,
    'Dairy, vegetable and eggs': 13,
    'Cheese, cold cuts, tapas': 14,
    'Meat, chicken, fish, vega': 15,
    'Salads, pizza, meals': 16,
    'Potatoes, vegetables, fruit': 17
}

adj_ids2 = {'Baby food': 0, 'Meat': 1, 'Chicken': 2, 'Fish': 3, 'Red wine': 4, 'White wine': 5, 'Rosé': 6,
       'Bake-off bread': 7, 'Pastries': 8, 'Dogs': 9, 'Cats': 10, 'Rodents': 11, 'Protein shakes': 12,
       'Protein bars': 13, 'Sports drinks': 14, 'Salads': 15, 'AH Fresh from home': 16, 'Fresh ready-to-eat meals': 17,
       'Processed meats': 18, 'Cheese': 19, 'Snacks': 20, 'Meal packs, mixes': 21,
       'Couscous, bulgur, quinoa, groats': 22, 'Pasta, rice, noodles': 23, 'Soups': 24, 'Preserves': 25, 'Spices': 26,
       'Frozen vegetables': 27, 'Ice cream': 28, 'Frozen snacks': 29, 'Sweet spreads': 30, 'Crackers, rice cakes': 31,
       'Breakfast cereals': 32, 'Milk': 33, 'Butter': 34, 'Yoghurt & cottage cheese': 35, 'Easter treats': 36,
       'Biscuits': 37, 'Sweets': 38, 'Soft drinks': 39, 'Tea': 40, 'Coffee': 41, 'Fruit biscuits & milk biscuits': 42,
       'Nut bars': 43, 'Fruit bars': 44, 'Beer': 45, 'Special beers': 46,
       'Alcohol-free beer and beer low in alcohol': 47, 'Fruit': 48, 'Vegetables': 49, 'Potatoes': 50}


def get_distance_matrix():
    """Function to get the distance matrix of the stores """
    return floyd_warshall(get_adj_matrix())


def get_adj_matrix():
    """ Creates an adjacency matrix from the adjacency list
    :return: adjacency matrix
    """
    adj_matrix = [[0] * len(categories) for _ in range(len(categories))]
    # loop over list
    for i in range(len(categories)):
        for j in range(len(categories)):
            if adj_matrix[i][j] != 0:
                continue
            elif categories[i] in adj_list and categories[j] in adj_list[categories[i]]:
                r = random.randint(20, 60)
                adj_matrix[i][j] = r
                adj_matrix[j][i] = r
    return adj_matrix


def floyd_warshall(graph):
    """ Function calculates all shortest paths to get the distances from all categories i to all categories j """
    n = len(graph)
    dist = [[graph[i][j] for j in range(n)] for i in range(n)]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != 0 and dist[k][j] != 0:
                    if dist[i][j] == 0 or dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
    mat = np.array(dist)
    n = mat.shape[0]
    mat[range(n), range(n)] = 15
    dist = list(mat)

    return dist


def print_matrix(s):
    """ Prints the adjacency matrix """
    print("     ", end="")
    for j in range(len(s[0])):
        print("%5d " % j, end="")
    print()
    print("     ", end="")
    for j in range(len(s[0])):
        print("------", end="")
    print()
    for i in range(len(s)):
        print("%3d |" % i, end="")
        for j in range(len(s[0])):
            print("%5d " % (s[i][j]), end="")
        print()
