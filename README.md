# Satirical News Detection
The objective of this project was to create a ML model with scikit-learn that can detect whether a news article is satirical based on its headline. The dataset used can be found [here](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection).

## The Process
First, the data was stored in a JSON file which had to be converted into a pandas dataframe.

```python
import json
import pandas as pd

def parse_data(file):
    for l in open(file, 'r'):
        yield json.loads(l)


data = list(parse_data('./Sarcasm_Headlines_Dataset_v2.json'))
df = pd.DataFrame(data)
```
After checking the amount of satirical and non-satirical headlines in the dataset, I found that they were roughly equal in number so no data re-balancing was required.

The data was split into training and testing sets in a 1:2 ratio. 
### TF-IDF Vectorization
In order to feed the headline text data into scikit-learn algorithms, the text must be converted into a vector.

TF-IDF is an abbreviation for "Term Frequency Inverse Document Frequency", and can essentially be thought of as the weighted frequency of each word in a document.

The term frequency (TF) vector describes how many times each word occurs in some text. However, if certain words are present in the majority of the text data, the presence of that word in a specfic document is less significant.

The inverse document frequency (IDF), is (as suggested by the name) a value that is higher the more infrequent a word is in the total dataset.

The TF-IDF vector is equal to the product of `TF X IDF`. From this it can be observed that a high TF-IDF value corresponds to high frequency of a term in a specific document but low frequency in the overall dataset. This [site](https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a) goes deeper into the math behind it.

In Python, it looks like this:
```python
tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
test_x_vector = tfidf.transform(test_x)
```
### Model Selection
I initially tried three different models, SVC, Decision Tree, and GaussianNB. After fitting each to the data, I used the score method of each to find the mean accuracy:
```markdown
SVC: 0.7920592906299629
Decision Tree: 0.7061937533086289
GNB: 0.6586553732133404
```

### Tuning the Model
The SVC model outperformed the other two, so I tried using GridSearchCV to find the optimal parameters.


```python
parameters = {'C': [1, 4, 8, 16, 32],'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(svc, parameters, cv=5)
grid_search.fit(train_x_vector, train_y)
print(grid_search.best_params_)
print(classification_report(test_y, grid_search.predict(test_x_vector)))
```
After ~30 to 40 minutes of runtime, was the output: `{'C': 8, 'kernel': 'rbf'}`.

#### Before
```markdown
              precision    recall  f1-score   support

           0       0.79      0.82      0.80      4916
           1       0.79      0.77      0.78      4529

    accuracy                           0.79      9445
   macro avg       0.79      0.79      0.79      9445
weighted avg       0.79      0.79      0.79      9445
```
#### After
```markdown
              precision    recall  f1-score   support

           0       0.80      0.83      0.81      4916
           1       0.81      0.77      0.79      4529

    accuracy                           0.80      9445
   macro avg       0.80      0.80      0.80      9445
weighted avg       0.80      0.80      0.80      9445
```
The f1-score improved by 1 percentage point.

While there was an improvement, there are some problems with GridSearchCV. One of them is that it only tests the parameters that you give it. For example, I only tested five different C values. The other problem is that it is very time-consuming. 

Nonetheless, I reran GridSearchCV with a narrower parameter range:
```python
parameters = {'C': [7, 7.5, 8, 8.5, 9], 'kernel': ['linear', 'rbf']}
```
