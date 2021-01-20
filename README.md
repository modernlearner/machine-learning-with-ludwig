Copyright (C) 2020-2021 Modern Learner Inc.

## Setup

Let's begin by installing virtual env and creating the environment:

    sudo pip install virtualenv
    virtualenv --python=python3.8 env

Then let's install the dependencies:

    source env/bin/activate
    pip install -r requirements.txt

Install English-specific encodings for spacy language processing:

    python -m spacy download en

Check for GPU support:

    python
    >>> from tensorflow.python.client import device_lib
    >>> device_lib.list_local_devices()

If you only see "CPU" in the devices list, follow the guide to ensure GPU support is enabled for Tensorflow: https://www.tensorflow.org/install/gpu

Run the code:

    python basic.py
    python tuning.py
    python digits.py

## Running the Numerai Example on Lambda Labs

1. launch an instance
2. use `scp` to copy `requirements.txt` and `numerai_example.py` to the instance
3. ssh into the instance
4. install the dependencies: `pip install -r requirements.txt --use-feature=2020-resolver`
5. check the number of gpus and gpu memory limit: `python -c 'from tensorflow.python.client import device_lib; device_lib.list_local_devices()'`
6. update the `gpu` and `gpu_memory_limit` configuration settings in the ludwig model
7. run the script with your numerai API credentials as environment variables
8. the model is saved, so you can `scp` it back to your local computer for backup

## Data Sources
The training and test set and the associated images for the MNIST dataset are sourced from here: https://www.kaggle.com/c/digit-recognizer/data

The Numerai data comes from their API and requires an account. You can download the latest training data after you login and upload your predictions.
