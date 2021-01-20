import pandas as pd
from ludwig.api import LudwigModel

import os.path

def open_dataset(url, storage_file, overwrite=False):
    if overwrite or not os.path.exists(storage_file):
        print('downloading dataset from', url)
        dataset = pd.read_csv(url)
        dataset.to_csv(storage_file)
        return dataset
    print('loading dataset from', storage_file)
    return pd.read_csv(storage_file)

input_features = [
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

model_config = {
    'input_features': input_features,
    'output_features': [
        {
            'name': 'target',
            'type': 'numerical',
        }
    ]
}

training_data = open_dataset(
    'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz',
    'datasets/numerai_training_data.csv'
)

model_kwargs = dict(gpus=[0], gpu_memory_limit=2000)
if os.path.exists('trained/numerai'):
    print('loading model')
    model = LudwigModel.load('trained/numerai', **model_kwargs)
    results, *other = model.evaluate(training_data)
    print('display results of evaluation')
    result = results['target']
    for key in result.keys():
      print('{}: {}'.format(key, result[key]))
else:
    print('creating the model')
    model = LudwigModel(model_config, **model_kwargs)
    print('training the model')
    results, *other = model.train(training_data)
    print('displaying results of training')
    result = results['training']['target']
    for key in result.keys():
      print('{}: {}'.format(key, result[key][0]))

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
