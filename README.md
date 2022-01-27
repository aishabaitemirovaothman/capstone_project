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

I wanted to check if the reaason why the models performed so poorly was because the target labels were not accurate. I figured that a good way to check would be to recreate my own labels using unsupervised learning models and rerun the models using those new labels and see if the performance improved. I used KMeans Clustering and Hierarchical clustering for creating new labels. With each model I tried creating first two then several clusters to see if the actual number of categories in the dataset was more than two.

<img width="410" alt="Screen Shot 2022-01-26 at 00 20 11" src="https://user-images.githubusercontent.com/92397144/151407019-c2521c8a-1468-4a8e-a283-69e42ac6249c.png">


## Evaluation

## Conclusion

## Recommendations

## Future Work

## For More Information

## Repository Structure


