# Sentiment Analysis Of Comments On Social Media
Author: Aisha Baitemirova - Othman

## Overview

Mental health issues is a big problem that is affecting the lives of millions of people around the world at any given moment. Problems like anxiety, depression, or bipolar disorder are just a few examples of mental health disorders that can severely damage people's lives if not addressed 

## Business Problem

One of the issues on social media is that it has a tendency to exponentially multiply the effects of its content. It's great if that content is nice and kind or educational and just positive in general. The problem comes when it is actually negative. People posting any kind of negative content on social media do usually need some inspiration and motivation to get and feel better. They need regular reminders that everything is going to be alright and that the world is a great place to be in. 

I decided to work on a sentiment analyzer that would be able to detect negative sentiments in a particular comment. I was hoping that the analyzer could be integrated into a social media algorithm and when someone posts something negative then the detector might flag that user and the user could be sent out some uplifting and inspirational ads or posts that would make them feel better. 

## Method

I performed all the steps of preprocessing my dataset to get it ready for feeding it into the machine learning models. I started with a dataset that contains 1.6 million comments from Twitter. First I performed the tokenization, lemmatization and stemming on the dataset. After that I got rid of all the words that appeared in the whole dataset less than 15 times thereby reducing the number of features that will be created by vectorization. The dataset was balanced with roughly half of the observations being labeled as negative, and the other half being labeled positive. 

<img width="418" alt="Screen Shot 2022-01-27 at 10 01 28" src="https://user-images.githubusercontent.com/92397144/151396196-533ca13f-44b2-491d-8d86-ddbea7fc3661.png">


## Results

## Evaluation

## Conclusion

## Recommendations

## Future Work

## For More Information

## Repository Structure


