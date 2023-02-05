pip install virtualenv
virtual env .cancer_detection
source .cancer_detection/bin/activate

pip install kaggle
https://www.kaggle.com/user/{your_username}/account # Get kaggle api credentials
Copy credentials to /home/{username}/.kaggle
kaggle competitions download -c rsna-breast-cancer-detection
mv rsna-breast-cancer-detection/ data/


