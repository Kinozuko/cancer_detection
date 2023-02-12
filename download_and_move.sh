#!/bin/bash

kaggle competitions download -c rsna-breast-cancer-detection

mkdir data

unzip rsna-breast-cancer-detection.zip -d data/