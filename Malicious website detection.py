
# Malicioue website detection using logistic Regression

#importing all the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#reading the csv file 
data = pd.read_csv(r"C:\Users\berni\PycharmProjects\Mallicious Url Detection\phishing_site_urls.csv")
data['category'] = data['Label']=='bad'

# importing seaborn and plotting the count of good and bad websites in the dataset
import seaborn as sns
sns.countplot(x = "Label",data = data)

from pandas import *
urls = data['URL'].tolist()
categories = data['category'].tolist()

# Vectorizing the urls using TfidVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
urls_vector = vectorizer.fit_transform(urls)
urls_vector[0]
urls_vector.shape

# dividing the dataset into testing and training data
train_url,test_url,train_category,test_category = train_test_split(urls_vector,categories,test_size = 0.3)

# creating the model for Logistic Regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver = 'saga')
model.fit(train_url,train_category)
acc = model.score(test_url,test_category) # gives accuracy of the model
print("Percentage Accuracy : ",acc*100,"%")

# predicting the values for the test dataset
predictions = model.predict(test_url) 

#getting user input to check the accuracy of the predicted data
num = int(input("enter the number : "))
c = ["Good Website","Bad Website"]
if (np.argmax(predictions[num])==test_category[num]):
  print("Correct Prediction : ",c[np.argmax(predictions[num])])

else:
  print(" Prediction : ",c[np.argmax(predictions[num])])
  print("Actual : ",c[test_category[num]])

#giving user input website as lists to validate our model 
p = ["kissasian.li","Google.com","www.yahoo.com","myuniversity.edu/renewal"]
p2 = vectorizer.transform(p)
for i in range(len(p)):
  if(model.predict(p2[i])==False):
    print(p[i],"is a Good Website")
  else:
    print(p[i],"is a Bad Website")

#Output of the model is :

# Percentage Accuracy :  95.51952622509162 %

# enter the number : 5
# Correct Prediction :  Good Website

# kissasian.li is a Bad Website
# Google.com is a Good Website
# www.yahoo.com is a Good Website
# myuniversity.edu/renewal is a Good Website