*** Main Simulation ***

"main.py":
- Runs the data simulation --> Outputs the generated datasets (purchases and fraud value)
- You can indicate the desired number of purchases (visits to the supermarket) and the location where you want the files to be stored
- When running the script the loading bars indicate the status of the program

"Purchases.py":
- Class that creates the supermarket purchases itself and introduces the fraud

"Customer.py":
- Class that represents one customer type

"product_catalog.py":
- It contains functions get the product files (from json files) and store them as a dataframe

"store_layout.py":
- It contains functions to create and get the adjacency matrix from the AH store



*** Data Directory ***

"product_data":
- It is a file system storing the products as individual text files
- Changes can be made by adding new files containing a product or by adjusting the attributes of existing products

"CustomerTypes":
- It stores the different customer types
- Changes can be made by adding new files containing a customer type or by adjusting the attribute values of existing files
- The mean and std can be adjusted to create slightly different versions of each customer type --> n indicates the number of versions

"Recipes.json":
- Contains all recipes in json (dictionary) format
- Changes can be made by adjusting or deleting or adding recipes.
- The recipes and their probabilities also need to be changed in the Customer Type Files



***  Generated_Datasets Directory ***

- Contains all kind of output data
- The purchase datasets: Contains all of the purchases generated with fraud introduced
- The fraud monetary datasets: Contain information about how much fraud (in monetary terms) are in each purchase
- The Product Catalog dataset: Contains all of the product information that are needed by the students.



*** Testing the performance ***

"PerformanceTesting":
- Contains multiple classification models
- The performance is printed for all of them (currently Accuracy and AUC and monetary fraud value)
- Models can easily be added and performance metrics changed

"lstm3":
- Contains the LSTM NN as one of the models



*** ProductCatalogFiles Directory ***

- You probably do not need these files
- Contain files that were used to process the original product dataset and store the products as json files