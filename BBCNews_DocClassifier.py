import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns 
#%matplotlib inline

import nltk
from nltk import FreqDist
from nltk.corpus import stopwords                   #Stopwords corpus
from nltk.stem import PorterStemmer                 # Stemmer

import re
import os

#Set up the working directory & retrieve the data files into a dataframe
data_folder = "./bbc"
news_folders = ["business","entertainment","politics","sport","tech"]

os.chdir(data_folder)

x = []
y = []
k=0
j=0
for i in news_folders:
    topic_files = os.listdir(i)
    k+=1
    for text_file in topic_files:
        file_path = i + "/" +text_file
        print ("reading file:", file_path)
        j+=1
        with open(file_path) as f:
            data = f.readlines()
            data = ' '.join(data)
        x.append(data)
        y.append(i)

print ("Total Folders read :", k) 
print ("Total files read :", j) 
  
data = {'NewsText': x, 'Category': y}       
BBCNews = pd.DataFrame(data)
#print 'writing csv flie ...'
#df.to_csv('../dataset.csv', index=False)
#DATA SET EXPLORATION
BBCNews['Category'].value_counts()
#sport            511
#business         510
#politics         417
#tech             401
#entertainment    386
#PLOT THE DISTRIBUTION OF TOPICS
#PLOT THE DISTRIBUTION ON NEWS TEXT 
BBCNews.NewsText.map(len).hist(figsize=(15, 5), bins=100)

#TEXT PRE PROCESSING
StopWords = set(stopwords.words('english')) 
snow = nltk.stem.SnowballStemmer('english')
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

def clean_text(txt):
       txt=txt.lower()
       cleanr = re.compile('<.*?>')
       txt = re.sub(cleanr, ' ', txt)  # remove html tags
       txt = re.sub('[^a-zA-Z#]',' ',txt) # Remove unwanted text 
       tokens=re.split('\W+',txt) 
       tokens = [item for item in tokens if item != ''] # remove null/whitespace tokens
       tokens = [lemmatizer.lemmatize(w) for w in tokens if w.lower() not in set(StopWords)]
       # filter out short tokens
       tokens = [w for w in tokens if len(w) > 3]       
       return tokens

# td-idf vectorizer
 from sklearn.feature_extraction.text import TfidfVectorizer
 bbc_tfidf = TfidfVectorizer(sublinear_tf=True,
                             norm='l2',
                             encoding='latin-1',
                             min_df=10,
                             ngram_range=(1,2),
                             stop_words='english',
                             
                             #preprocessor = clean_text(),
                             analyzer =clean_text
                             )
bbc_tfidf_counts=bbc_tfidf.fit_transform(BBCNews['NewsText'])
print("Tf-Idf sparse matrix has {} rows and {} columns".format(bbc_tfidf_counts.shape[0],bbc_tfidf_counts.shape[1]))

bbc_tfidf_df = pd.DataFrame(bbc_tfidf_counts.toarray())

bbc_tfidf.get_feature_names()
y_labels = BBCNews['Category']
features = bbc_tfidf_counts.toarray()

# MODEL EVALUATION
     
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC,LinearSVC


models = [RandomForestClassifier(n_estimators=250, max_depth=5, random_state=145),
          MultinomialNB(),
          LogisticRegression(random_state=789),
          SVC(),
          LinearSVC(multi_class='ovr', penalty='l2')]
       
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, y_labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


#MODEL ACCURACY MEAN
cv_df.groupby('model_name').accuracy.mean()
# Model interpretation LOGISTIC REGRESSION
from sklearn.model_selection import train_test_split

model = LogisticRegression(random_state=0)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, y_labels,bbc_tfidf_df.index, test_size=0.30, random_state=125)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)


from sklearn.metrics import confusion_matrix,accuracy_score,cohen_kappa_score

conf_mat = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred) #98.20%
kap = cohen_kappa_score(y_test, y_pred) 

print("Model accuracy: {} and kappa: {}".format(round(acc*100,2),round(kap,2)))
