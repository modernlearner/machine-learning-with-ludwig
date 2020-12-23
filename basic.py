# Disable output of logging messages from TensorFlow if it's too noisy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os.path

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

def load_or_create_model(model_dir, model_config, **model_kwargs):
    if os.path.exists(model_dir):
        print('Loading the model: {}'.format(model_dir))
        return (LudwigModel.load(model_dir, **model_kwargs), True)
    else:
        print('Defining the model')
        return (LudwigModel(model_config, **model_kwargs), False)
        
base_model, base_model_loaded = load_or_create_model(
    'trained/basic', base_model_definition, gpus=[0], gpu_memory_limit=2000
)
if base_model_loaded:
    training_stats, preprocessed_data, output_directory = base_model.evaluate(
        dataset='./datasets/spam_test.csv'
    )
else:
    print('Training the model')
    training_stats, preprocessed_data, output_directory = base_model.train(
        training_set='./datasets/spam_train.csv',
        test_set='./datasets/spam_test.csv',
        skip_save_processed_input=True,
    )
    base_model.save('trained/basic')
    
print(training_stats)
"""
visualize.learning_curves(
    [{ 'training': training_stats }],
    None
)
visualize.confusion_matrix(
    test_stats_per_model={ 'base_model': training_stats['spam'] },
    metadata=preprocessed_data,
    output_feature_name=None,
    top_n_classes=[3],
    normalize=True,
)
visualize.frequency_vs_f1(
    test_stats_per_model=training_stats,
    metadata=preprocessed_data,
    output_feature_name='spam',
    top_n_classes=[3],
)
"""

print('Creating a 2nd model classifier with some adjustments')
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
        'activation': 'relu', # Activation used for *all* layers.
        'num_filters': 32,
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

other_model, other_model_loaded = load_or_create_model(
    'trained/basic_other', other_model_definition, gpus=[0], gpu_memory_limit=2000
)
if other_model_loaded:
    training_stats, preprocessed_data, output_directory = other_model.evaluate(
        dataset='./datasets/spam_test.csv'
    )
else:
    print('Training the model')
    training_stats, preprocessed_data, output_directory = other_model.train(
        training_set='./datasets/spam_train.csv',
        test_set='./datasets/spam_test.csv',
        skip_save_processed_input=True,
    )
    other_model.save('trained/basic_other')
print(training_stats)
# Comparing the testing results of the models
# visualize.compare_performance(
#     test_stats_per_model=[test_result, other_model_test_result],
#     output_feature_name='spam',
#     model_names=['Base Model', 'Other Model'],
# )

def print_predictions(unpredicted_emails, model):
    prediction_result, output_directory = model.predict(dataset=unpredicted_emails)
    emails = unpredicted_emails.join(prediction_result)
    for index, row in emails.iterrows():
        print('{} ({:.6f}): {} / {}'.format(
            row.get('spam_predictions'),
            row.get('spam_probability'),
            row.get('subject')[0:30],
            row.get('content')[0:30],
        ))

unpredicted_emails = pd.read_csv('./datasets/spam_unpredicted.csv')
print('Prediction Results for Base Model')
print_predictions(unpredicted_emails, base_model)
print('Prediction Results for Other Model')
print_predictions(unpredicted_emails, other_model)
