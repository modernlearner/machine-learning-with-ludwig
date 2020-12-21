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

Run the code:

    python basic.py
    python tuning.py
    python digits.py

## Data Sources
The training and test set and the associated images for the MNIST dataset are sourced from here: https://www.kaggle.com/c/digit-recognizer/data
