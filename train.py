# train.py

from model.support_vector.support_vector_regression import train_svr

# Run the training function
print('-'*50)
print('Start gradient_boosting training_hp')
train_svr()
print('Training is over')


# from model.random_forest_classification.random_forest_classification import train_and_evaluate_classification

# # Run the classification training and evaluation
# train_and_evaluate_classification()