import os.path
from sys import argv, exit

import pandas as pd
from ludwig.api import LudwigModel
from ludwig import visualize

def open_dataset(url, storage_file, overwrite=False):
    if overwrite or not os.path.exists(storage_file):
        print('downloading dataset from', url)
        dataset = pd.read_csv(url)
        dataset.to_csv(storage_file)
        return dataset
    print('loading dataset from', storage_file)
    return pd.read_csv(storage_file)

INPUT_FEATURES = [
    {
        'name': 'feature_intelligence{}'.format(i + 1),
        'type': 'numerical',
    } for i in range(12)
] + [
    {
        'name': 'feature_charisma{}'.format(i + 1),
        'type': 'numerical',
    } for i in range(86)
] + [
    {
        'name': 'feature_strength{}'.format(i + 1),
        'type': 'numerical',
    } for i in range(38)
] + [
    {
        'name': 'feature_dexterity{}'.format(i + 1),
        'type': 'numerical',
    } for i in range(14)
] + [
    {
        'name': 'feature_constitution{}'.format(i + 1),
        'type': 'numerical',
    } for i in range(114)
] + [
    {
        'name': 'feature_wisdom{}'.format(i + 1),
        'type': 'numerical',
    } for i in range(46)
]

def create_model_config(additional_output_feature_options):
    return {
        'input_features': INPUT_FEATURES,
        'output_features': [
            {
                'name': 'target',
                'type': 'numerical',
                **additional_output_feature_options
            }
        ]
    }

def load_or_create_model(model_id, model_config):
    model_kwargs = dict(gpus=[0], gpu_memory_limit=2500)
    model_path = 'trained/numerai_{}'.format(model_id)
    if os.path.exists(model_path):
        print('loading model #{}'.format(model_id))
        model = LudwigModel.load(model_path, **model_kwargs)
        results, *other = model.evaluate(TRAINING_DATA)
        print('display results of evaluation for model #{}'.format(model_id))
        result = results['target']
        for key in result.keys():
          print('{}: {}'.format(key, result[key]))
    else:
        print('creating model #{}'.format(model_id))
        model = LudwigModel(model_config, **model_kwargs)
        print('training model #{}'.format(model_id))
        results, *other = model.train(TRAINING_DATA)
        print('displaying results of training for model #{}'.format(model_id))
        result = results['training']['target']
        for key in result.keys():
          print('{}: {}'.format(key, result[key][0]))
        print('saving model #{}'.format(model_id))
        model.save(model_path)
    return model, results

def generate_and_save_predictions(model):
    tournament_data = open_dataset(
        'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz',
        'datasets/numerai_tournament_data.csv'
    )
    PREDICTION_CSV = './numerai_result.csv'
    if os.path.exists(PREDICTION_CSV):
        print('loading already generated predictions')
        predictions = pd.read_csv(PREDICTION_CSV)
    else:
        print('generating predictions using model')
        predictions, *other = model.predict(tournament_data)
        result = tournament_data.join(predictions)
        print('saving prediction results to', PREDICTION_CSV)
        result.to_csv(PREDICTION_CSV)
        predictions = result

    predictions_df = tournament_data['id'].to_frame()
    predictions_df['prediction_kazutsugi'] = predictions['target'].to_frame()
    # TODO: need to figure out why there are some empty values instead of just filling them in with 0.0
    predictions_df.fillna(value=0.0, inplace=True)
    print(predictions_df.head())

    print('Numerai predictions saved, login to the website and upload them')
    predictions_df.to_csv('./numerai_predictions.csv', index=False)

# See: https://ludwig-ai.github.io/ludwig-docs/user_guide/#numerical-output-features-and-decoders
MODEL_CONFIGS = [
    {
        'fc_layers': [
            { 'fc_size': 64 },
            { 'fc_size': 64 },
        ],
        'num_fc_layers': 2,
        'norm': 'batch',
    },
    {
        'fc_layers': [
            { 'fc_size': 128 },
            { 'fc_size': 64 },
        ],
        'num_fc_layers': 2,
        'norm': 'batch',
    },
    {
        'fc_layers': [
            { 'fc_size': 128, 'activation': 'tanh', },
            { 'fc_size': 64, 'activity_regularizer': 'l1' },
            { 'fc_size': 32 },
        ],
        'num_fc_layers': 3,
        'norm': 'batch',
    },
]

def print_usage():
    print('Usage:')
    print('    python numerai_multiple.py train')
    print('Check the results and then select the model to use for the predictions')
    print('    python numerai_multiple.py predict 0')

if len(argv) < 2:
    print_usage()
    exit(1)

if argv[1] == 'train':
    TRAINING_DATA = open_dataset(
        'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz',
        'datasets/numerai_training_data.csv'
    )
    MODELS = [
        load_or_create_model(i, create_model_config(MODEL_CONFIGS[i]))
        for i in range(len(MODEL_CONFIGS))
    ]

    visualize.compare_performance(
        test_stats_per_model=[stats for model, stats in MODELS],
        output_feature_name='target',
        model_names=['Model {}'.format(i) for i in range(len(MODELS))],
    )
elif argv[1] == 'predict':
    TRAINING_DATA = open_dataset(
        'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz',
        'datasets/numerai_training_data.csv'
    )
    model_id = int(argv[2])
    if model_id < 0 or model_id >= len(MODEL_CONFIGS):
        print('model id must be between 0 and {} (inclusive')
        exit(2)
    model, results = load_or_create_model(model_id, create_model_config(MODEL_CONFIGS[model_id]))
    generate_and_save_predictions(model)
else:
    print_usage()
    exit(1)
