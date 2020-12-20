# Disable output of logging messages from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import pandas as pd
from ludwig.api import LudwigModel
from ludwig import visualize

# Definition of input features
input_features = [
    {
        'name': 'subject',
        'type': 'text',
        'preprocessing': {
            'lowercase': True,
        },
    },
    {
        'name': 'content',
        'type': 'text',
        'preprocessing': {
            'lowercase': True,
        },
        'encoder': 'stacked_parallel_cnn',
        'reduce_output': 'attention',
        'activation': 'relu',
    },
]

# Definition of output features
output_features = [
    {
        'name': 'spam',
        'type': 'category',
    }
]

base_model_definition = {
    'input_features': input_features,
    'output_features': output_features,
}

# Defining and training the model
base_model = LudwigModel(base_model_definition)
training_stats = base_model.train(data_csv='./datasets/spam_train.csv')

# Testing the model on data with known output
test_result = base_model.test(data_csv='./datasets/spam_test.csv')[1]
"""
visualize.confusion_matrix(
    test_stats_per_model=[test_result],
    metadata=base_model.train_set_metadata,
    output_feature_name='spam',
    top_n_classes=[3],
    normalize=True,
)
visualize.frequency_vs_f1(
    test_stats_per_model=[test_result],
    metadata=base_model.train_set_metadata,
    output_feature_name='spam',
    top_n_classes=[3],
)
"""

# Creating a 2nd model classifier with some adjustments
other_model_definition = base_model_definition.copy()
other_model_definition['input_features'] = [
    {
        'name': 'subject',
        'type': 'text',
        'preprocessing': {
            'lowercase': True,
        },
    },
    {
        'name': 'content',
        'type': 'text',
        'preprocessing': {
            'lowercase': True,
        },
        'encoder': 'stacked_parallel_cnn',
        'reduce_output': 'attention',
        'activation': 'tanh', # Activation used for *all* layers.
                              # If 'tanh' is used here, the runtime is ~33 seconds.
                              # If 'relu' is used, runtime drops to ~15 seconds.
                              # However, with 'tanh', training accuracy is higher.
        'num_filters': 51,
        'stacked_layers': [
            [
                { 'filter_size': 4 },
                { 'filter_size': 5 },
                { 'filter_size': 3 },
            ],
            [
                { 'filter_size': 2 },
                { 'filter_size': 2 },
                { 'filter_size': 2 },
            ],
            [
                { 'filter_size': 3 },
                { 'filter_size': 4 },
                { 'filter_size': 5 },
            ]
        ],
    },
]

other_model = LudwigModel(other_model_definition)
other_model.train(data_csv='./datasets/spam_train.csv')
other_model_test_result = other_model.test(data_csv='./datasets/spam_test.csv')[1]

# Comparing the testing results of the models
visualize.compare_performance(
    test_stats_per_model=[test_result, other_model_test_result],
    output_feature_name='spam',
    model_names=['Base Model', 'Other Model'],
)

def print_predictions(unpredicted_emails, model):
    prediction_result = model.predict(data_df=unpredicted_emails)
    emails = unpredicted_emails.join(prediction_result)
    for index, row in emails.iterrows():
        print('{} ({:.6f}): {} / {}'.format(
            row.get('spam_predictions')[:4],
            row.get('spam_probability'),
            row.get('subject')[0:30],
            row.get('content')[0:30],
        ))

unpredicted_emails = pd.read_csv('./datasets/spam_unpredicted.csv')
print('Prediction Results for Base Model')
print_predictions(unpredicted_emails, base_model)
print('Prediction Results for Other Model')
print_predictions(unpredicted_emails, other_model)
