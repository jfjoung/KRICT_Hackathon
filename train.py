# train.py

from model.gradient_boosting_regressor.gradient_boosting import train_gradient_boosting

# Run the training function
print('-'*50)
print('Start random forest training')
train_gradient_boosting()
print('Training is over')


# from model.random_forest_classification.random_forest_classification import train_and_evaluate_classification

# # Run the classification training and evaluation
# train_and_evaluate_classification()