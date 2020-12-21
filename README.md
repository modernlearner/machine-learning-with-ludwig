Copyright (C) 2020 Modern Learn Inc.

## Setup

Install virtual env and create the environment:

    sudo pip install virtualenv
    virtualenv --python=python3.8 env

Install the dependencies:

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

## Data Sources
The training and test set and the associated images for the MNIST dataset are sourced from here: https://www.kaggle.com/c/digit-recognizer/data
