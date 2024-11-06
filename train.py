# train.py

from model.random_forest.random_forest import train_random_forest

# Run the training function
print('-'*50)
print('Start random forest training')
train_random_forest()
print('Training is over')


# from model.random_forest_classification.random_forest_classification import train_and_evaluate_classification

# # Run the classification training and evaluation
# train_and_evaluate_classification()