# Disable output of logging messages from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import pandas as pd
from ludwig.api import LudwigModel
from ludwig import visualize

input_features = [
    {
        'name': 'pixel{}'.format(i),
        'type': 'numerical',
    } for i in range(784)
]

base_model_definition = {
    'input_features': input_features,
    'output_features': [
        {
            'name': 'label',
            'type': 'category',
        }
    ],
}

base_model = LudwigModel(base_model_definition)
dataset = pd.read_csv('./datasets/digits_train.csv')
number_of_rows = len(dataset.index)
ratio = 0.75
training_boundary = round(number_of_rows * ratio)

training_data = dataset.iloc[:training_boundary, :]
test_data = dataset.iloc[training_boundary:, :]

print(training_data.head())
print(test_data.head())

training_stats = base_model.train(training_set=training_data, test_set=test_data)

visualize.compare_performance(
    test_stats_per_model=[test_result],
    output_feature_name='label',
    model_names=['model'],
)
