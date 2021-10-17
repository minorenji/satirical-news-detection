"""
Sean Lim
10/16/2021

A machine learning model that classifies news headlines into 'satirical' or 'not satirical' categories
"""

import pandas as pd
import json
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix


# read JSON file into pandas dataframe
def parse_data(file):
    for l in open(file, 'r'):
        yield json.loads(l)


data = list(parse_data('./Sarcasm_Headlines_Dataset_v2.json'))
df = pd.DataFrame(data)

train, test = train_test_split(df, test_size=0.33, random_state=42)
train_x, train_y = train['headline'], train['is_sarcastic']
test_x, test_y = test['headline'], test['is_sarcastic']

tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
test_x_vector = tfidf.transform(test_x)

# SVC
svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)

print("SVC: " + str(svc.score(test_x_vector, test_y)))

print(classification_report(test_y, svc.predict(test_x_vector)))

print(confusion_matrix(test_y, svc.predict(test_x_vector)))

# Decision Tree
dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)

print("Decision Tree: " + str(dec_tree.score(test_x_vector, test_y)))

# Naive Bayes
gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

print("GNB: " + str(gnb.score(test_x_vector.toarray(), test_y)))

'''
print(svc.predict(tfidf.transform(
    ['Intergalactic Animal Rights Groups Condemn Use Of Brutal, Unsanitary Planet To Raise Human Meat'])))
print(svc.predict(tfidf.transform(['Most Terrifying Ways The Government Is Spying On You'])))
print(svc.predict(tfidf.transform(['Remington Introduces Ammunition For Sensitive Skin'])))
'''