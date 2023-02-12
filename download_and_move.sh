#!/bin/bash

kaggle competitions download -c rsna-breast-cancer-detection

unzip rsna-breast-cancer-detection.zip

mv rsna-breast-cancer-detection/ data/
