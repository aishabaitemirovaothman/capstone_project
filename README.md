# Sentiment Analysis Of Comments On Social Media
Author: Aisha Baitemirova - Othman

## Overview

Mental health issues is a big problem that is affecting the lives of millions of people around the world at any given moment. Problems like anxiety, depression, or bipolar disorder are just a few examples of mental health disorders that can severely damage people's lives if not addressed 

## Business Problem

One of the issues on social media is that it has a tendency to exponentially multiply the effects of its content. It's great if that content is nice and kind or educational and just positive in general. The problem comes when it is actually negative. People posting any kind of negative content on social media do usually need some inspiration and motivation to get and feel better. They need regular reminders that everything is going to be alright and that the world is a great place to be in. 

I decided to work on a sentiment analyzer that would be able to detect negative sentiments in a particular comment. I was hoping that the analyzer could be integrated into a social media algorithm and when someone posts something negative then the detector might flag that user and the user could be sent out some uplifting and inspirational ads or posts that would make them feel better. 

## Method

I performed all the steps of preprocessing my dataset to get it ready for feeding it into the machine learning models. I started with a dataset that contains 1.6 million comments from Twitter. First I performed the tokenization, lemmatization and stemming on the dataset as well as removal of stopwords. After that I got rid of all the words that appeared in the whole dataset less than 15 times thereby reducing the number of features that will be created by vectorization. The dataset was balanced with roughly half of the observations being labeled as negative, and the other half being labeled nonnegative. 

<img width="418" alt="Screen Shot 2022-01-27 at 10 01 28" src="https://user-images.githubusercontent.com/92397144/151396196-533ca13f-44b2-491d-8d86-ddbea7fc3661.png"> 

I looked into the duplicates in each category of comments. And here are the phrases that most commonly appeared among the non-negative comments vs the negative comments respectively:

<img width="435" alt="Screen Shot 2022-01-25 at 22 21 15" src="https://user-images.githubusercontent.com/92397144/151403316-4c7be66c-bc2d-4ab6-a8f1-170f46e93b7c.png"> 
<img width="441" alt="Screen Shot 2022-01-25 at 22 20 14" src="https://user-images.githubusercontent.com/92397144/151403334-b764d9a9-393f-41c4-abeb-d937bb13b18a.png">


After that I performed a vectorization using TfidfVectorizer and the train - test split of the resulting feature vectors. At first I tried the vectorization on the whole dataset, but I encountered some technical issues because of the computing limitations on my laptop due to RAM not being enough so I had to trim the dataset all the way to 600 000 observations. In the vectorizer I specified the parameters min_df and max_df because I only wanted to keep the terms that were within a specific window of frequency in the document. That way I ended up with 619 features that I used to feed into TruncatedSVD of sklearn to perform dimensionality reduction. As a result I ended up with 300 features. 

## Results

The models that I used for prediction were : Gaussian Naive Bayes (baseline model), Logistic Regression, Decision Tree Classifier, Random Forest Classifier, and Neural Networks. The results that I obtained were unfortunately not satisfactory enough. 

Gaussian Naive Bayes (baseline model) accuracy score: 0.40763636363636363
Logistic Regression: 0.6265454545454545
Decision Tree Classifier: 0.6253333333333333
Random Forest Classifier: 0.5142424242424243
Neural Networks (150 epochs): 0.5259

I used GridSearch CV on all the models except for the baseline model and the results did not improve significantly. 

I wanted to check if the reaason why the models performed so poorly was because the target labels were not accurate. I figured that a good way to check would be to recreate my own labels using unsupervised learning models and rerun the models using those new labels and see if the performance improved. I used KMeans Clustering and Hierarchical clustering for creating new labels. With each model I tried creating first two then several clusters to see if the actual number of categories in the dataset was more than two. Looking at the elbow graph it looks like increasing the number of labels would not help that much because the sum of squared distances did not drop significantly enough with each additional cluster. 

<img width="410" alt="Screen Shot 2022-01-26 at 00 20 11" src="https://user-images.githubusercontent.com/92397144/151407019-c2521c8a-1468-4a8e-a283-69e42ac6249c.png">

I ran the Logistic Regression model again using the new target labels created by the unsupervised models, and unfortunately the accuracy score that I obtained dropped to 40 percent (down from 60 percent the first time). 


## Evaluation

I have several possible explanations as to why the accuracy scores of the models were so low. The first possibility is that the dataset was not labeled properly. The second possible explanation is that even after all the preprocessing steps taken, there is still some noise left in the dataset that needs to be detected and removed before running the models again. Another possible explanation is that the vectorizer that I used did not pick up the most important words properly. I am hoping that trying new vectorizers such as HashingVectorizer or CountVectorizer could help. 

## Conclusion

My conclusion is that the detector is not ready to be deployed yet if one were to focus on the accuracy score. However, if we were to focus on recall score instead of the accuracy score than the picture looks different. In the case of this detector we prefer to make type I Error, meaning that it is better to identify someone as posting negative comments and send him/her inspirational ads/posts even if in reality their comments were not negative. It is preferable to having type II Error and ignore someone who is posting negative comments and that person would miss on the benefits of being regularly exposed to some positive content sent out by the social media algorithm to those who were flagged as posting negative comments. In this case we would focus on recall score because it measures how good a particular machine learning model is at identifying true positives. The decision tree classifier that I ran after adjusting the hyperparameters had a recall score of 96 percent. That means that the model correctly identified the true positives (in this case the negative comments) 96 percent of the time. So the readiness of the prediction model depends on whether one wants to focus on the accuracy score or the recall score. 

## Future Work

I will keep working on improving the results of my models so I have some steps in mind that I want to take in order to raise the accuracy score:

1) Try a new vectorizer that would identify most important words in the document more efficiently. 
2) Try other types of Neural Networks.
3) Use Word Embedding.

Whenever I reach some optimal results I would like to go further and work on making predictions on depression, anger or sarcasm.

## For More Information

Please have a look at my code and my presentation.

## Repository Structure

├── [data]
├── [images]
├── .gitignore
├── README.md
└── notebook.ipynb
