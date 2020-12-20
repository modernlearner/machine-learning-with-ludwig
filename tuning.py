# Re-visting basic.py, this time with parameter tuning

# Disable output of logging messages from TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from datetime import datetime
import pandas as pd
from ludwig.api import LudwigModel
from ludwig import visualize

def make_model(encoder_tuning):
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
            **encoder_tuning,
        },
    ]
    model_definition = {
        'input_features': input_features,
        'output_features': [{ 'name': 'spam', 'type': 'category', }]
    }
    return LudwigModel(model_definition)

training_df = pd.read_csv('./datasets/spam_train.csv')
test_df = pd.read_csv('./datasets/spam_test.csv')
test_results = []
model_names = []
print('num_filters, first_layer_filter_sizes, second_layer_filter_sizes, third_layer_filter_sizes, activation, reduce_output')
for num_filters in [15, 51]:
    for first_layer_filter_sizes in [[4,5,3]]:
        for second_layer_filter_sizes in [[2,2,2]]:
            for third_layer_filter_sizes in [[3,4,5]]:
                for activation in ['relu', 'tanh']:
                    for reduce_output in ['attention', 'concat']:
                        model = make_model({
                            'reduce_output': reduce_output,
                            'activation': activation,
                            'num_filters': num_filters,
                            'stacked_layers': [
                                [{ 'filter_size': filter_size } for filter_size in first_layer_filter_sizes],
                                [{ 'filter_size': filter_size } for filter_size in second_layer_filter_sizes],
                                [{ 'filter_size': filter_size } for filter_size in third_layer_filter_sizes],
                            ],
                        })
                        start = datetime.utcnow()
                        model.train(data_df=training_df)
                        after_training = datetime.utcnow()
                        test_result = model.test(data_df=test_df)[1]
                        after_testing = datetime.utcnow()
                        print('Testing Results for Model: {}, {}, {}, {}, {}, {}'.format(
                            num_filters,
                            first_layer_filter_sizes, second_layer_filter_sizes, third_layer_filter_sizes,
                            activation,
                            reduce_output,
                        ))
                        print('Total time:', after_testing - start)
                        test_results.append(test_result)
                        model_names.append('Model: {}, {}, {}, {}, {}, {}'.format(
                            num_filters,
                            first_layer_filter_sizes, second_layer_filter_sizes, third_layer_filter_sizes,
                            activation,
                            reduce_output,
                        ))

visualize.compare_performance(
    test_stats_per_model=test_results,
    output_feature_name='spam',
    model_names=model_names,
)
