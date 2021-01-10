from ludwig.api import LudwigModel

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
# TODO: load existing model and use evaluate instead of training (good for future submissions to Numerai if the model is okay)
model = LudwigModel(model_config, gpus=[0], gpu_memory_limit=3000)
results, *other = model.train(dataset='./datasets/numerai_training_data.csv', skip_save_processed_input=True)
model.save('trained/numerai')
print(results)
results, *other = model.predict(dataset='./datasets/numerai_tournament_data.csv')
print(results)
# TODO: save the results to a CSV in the correct format