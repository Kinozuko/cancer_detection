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
```

# Execute process to convert dcm to png

```
Execute python main.py --method process --n_pools 2
Arguments:
    --method List of values [process]
    --n-pool Integer greater to zero
```