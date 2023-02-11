# How to create virtualenv

```
pip install virtualenv
virtualenv .cancer_detection
source .cancer_detection/bin/activate
pip install -r requirements.txt 
```

# How to download dataset from Kaggle

```
pip install kaggle
https://www.kaggle.com/{your_username}/account
Go to API section and Create New API Token to download the credentials for your account
Copy credentials to /home/{username}/.kaggle
kaggle competitions download -c rsna-breast-cancer-detection
mv rsna-breast-cancer-detection/ data/
```

# Execute process to convert dcm to png

```
Execute python main.py --method process --n_pools 2
Arguments:
    --method List of values [process]
    --n_pool Integer greater to zero
```