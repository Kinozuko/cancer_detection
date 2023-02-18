# Pre-requisities
```
## Open CV

apt-get install libgl1-mesa-glx
apt-get install libglib2.0-dev

```

# How to create virtualenv

```
pip install virtualenv
virtualenv .cancer_detection
source .cancer_detection/bin/activate
pip install -r requirements.txt 
```

# How to download dataset from Kaggle

```
https://www.kaggle.com/{your_username}/account
Go to API section and Create New API Token to download the credentials for your account

Copy credentials to /home/{username}/.kaggle

Now protect our API key using chmod 600 /home/{username}/.kaggle (remember the path can be different for you)

Give execution permissions to download_and_move.sh:

chmod +x download_and_move.sh

./download_and_move.sh

NOTE: Be aware you have installed unzip in your system
```

# Execute process to convert dcm to png

```
Execute python main.py --method process --n_pools 2

Arguments:
    --method List of values [process, "train"]
    --n-pool Number of pools, greater to zero
```

# Execute process to train model

```
Execute python main.py --method train --model-version v1 --n-batch 30 --n-epoch 1 --patience 5

Arguments:
    --method List of values [process "train"]
    --model-version Version of the model to run [v1, v2]
    --n-batch Number of batchs, greater to zero
    --n-epoch Number of epochs, greater to zero
    --patience Number of patience to use in Early Stopping, greater to zero
```