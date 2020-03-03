This repository is dedicated to the Movie recommendation system which designed and developed by Python3 and the data which used in this project is based on the provided dataset by Kaggle, You could find the required dataset at https://www.kaggle.com/rounakbanik/the-movies-dataset .  

The first thing for this project is knowing the dataset that we need, among all csv files which provided by Kaggle all we need is credits.csv, keywords.csv, links.csv, links_small.csv, movies_metadata.csv, ratings_small.csv

This code uses demographic filtering and content filtering to recommend movies on the web page. Flask API is used to deploy model on website.

Here is the list of dependecies which used in this project and how you could them:

pip install numpy
pip install pandas
pip install matplotlib
pip install scipy
pip install ast # this dependecy is already covered by python3
pip install sklearn
pip install nltk
pip install flask
pip install warnings # this dependecy is already covered by python3

Onece the dependencies are installed properly you could run the webapp file with this command python3 webapp.py. 

![Screenshot_2020-03-03 Recommender System(1)](https://user-images.githubusercontent.com/23243761/75825450-aeb05480-5da5-11ea-885a-31876a9b220e.png)

This recommendation system is working based on three categories which are based on popularity,genere and content

![Screenshot_2020-03-03 Recommender System](https://user-images.githubusercontent.com/23243761/75824500-e61e0180-5da3-11ea-86fa-e1044a212524.png)
