# Twitter Sentiment Analysis for Google and Apple Products
## Project Summary
This project focuses on performing sentiment analysis on tweets related to Apple and Google products. Leveraging a dataset of over 9,000 human-rated tweets, the goal is to classify sentiments as positive, negative, or neutral to understand public perception and identify trends. The project aims to build and evaluate multiple Natural Language Processing (NLP) models, including Logistic Regression, Multinomial Naive Bayes, K-Nearest Neighbor, Random Forest Classifier, and Neural Networks, to predict both the product an emotion is directed at and the underlying sentiment. The best-performing models will be integrated into a Streamlit deployment application for real-time insights.

## Data Understanding
The dataset used is tweet_product_company.csv, which contains real-world tweet data mentioning Apple and Google products. Each entry includes the tweet text, the referenced product (e.g., iPhone, Google), the associated company (Apple or Google), and a sentiment label (positive, negative, or neutral). The dataset is valuable due to the short, noisy, and opinion-driven nature of tweets, making it ideal for testing robust NLP techniques. It supports both binary and multiclass sentiment prediction and offers real-world variability, including emoji usage, slang, and abbreviations.

## Problem Statement
In today's digital age, social media serves as a vast platform for customers to express their opinions. For major tech brands like Apple and Google, understanding these sentiments is crucial. Sentiment analysis provides deep insights into public perception, allowing businesses to gauge customer emotions and proactively address concerns, ultimately leading to improved products and services.

## Business Objectives
The primary client for this NLP project is Apple. By analyzing sentiments from tweets about their products and those of their competitors, Apple can gain authentic feedback that traditional methods might miss. This real-time access to customer sentiment will enable Apple to quickly identify trends, preferences, and potential issues, facilitating proactive engagement and timely adjustments to their strategies.

## Project Objectives:
Identify the distribution of positive, negative, and neutral sentiments by company (Apple vs. Google).
The initial analysis of the dataset revealed the distribution of sentiments across Apple and Google products. After cleaning and categorizing products, the sentiment distribution showed that a significant portion of tweets were neutral, followed by positive, and then negative sentiments. This highlights the importance of distinguishing between no emotion and actual positive/negative feedback.
[Insert picture: sentiment-distribuction-per-product.png]

Build, train, and tune a Logistic Regression model for categorizing whether the sentiment in a tweet is directed to an Apple or a Google product.
A Logistic Regression model will be developed to classify tweets as pertaining to either Apple or Google products. This model will help in understanding which company's products are being discussed in a given tweet, serving as a preliminary classification step before sentiment prediction.

Build, train, and tune four classification models (Multinomial Naive Bayes, K-Nearest Neighbour, Random Forest Classifier, Neural Network) for predicting the underlying sentiment in a tweet.
Four distinct classification models will be implemented to predict sentiment (positive, negative, neutral). These models will be trained and tuned to optimize their performance on the preprocessed tweet data.

Evaluate and compare the performance of the four sentiment models to select the most robust and generalizable alternative for deployment.
The performance of each sentiment model will be rigorously evaluated using metrics such as Accuracy, Precision, Recall, F1-score, and Confusion Matrices. This comparative analysis will determine the most effective model for deployment, ensuring it is robust and generalizable to new, unseen tweet data.

Build a Streamlit deployment app that integrates the Logistic Regression model (for predicting the product the emotion is directed at) and the selected sentiment prediction model (for predicting the underlying sentiment in a tweet).
A user-friendly Streamlit application will be developed to demonstrate the utility of the trained models. This app will allow users to input tweets and receive predictions on both the associated product (Apple or Google) and the sentiment (positive, negative, or neutral), providing a practical tool for real-time sentiment analysis.

## Conclusion
This project successfully demonstrates the application of NLP techniques for sentiment analysis on social media data. By meticulously cleaning and preprocessing the tweet dataset, and then training and evaluating various classification models, we can effectively gauge public opinion towards Apple and Google products. The chosen models provide valuable insights into consumer sentiment, which can be leveraged for strategic business decisions.

## Recommendations
**Focus on Negative Sentiment Analysis:** Given the lower count of negative tweets, consider oversampling negative instances or using weighted loss functions during model training to improve the detection of critical feedback.

**Product-Specific Sentiment:** Further analysis could delve into sentiment towards specific products (e.g., "iPhone 15" vs. "Google Pixel 8") rather than just the company, providing more granular insights.

**Temporal Analysis:** Incorporate time-series analysis to track sentiment trends over time, especially around product launches or major company announcements, to identify immediate public reactions.

## Next Steps
1. Deploy the FastAPI backend to a Cloud Service.

2. Incorporate real-time Tweet streaming via the X API to stream tweets filtered by keywords.

3. Build a Tableau Dashboard to visualize sentiment and product trends in real-time to support data-informed decisions.

4. Experiment with Transfer Learning by leveraging pretrained transformers for NLP, such as BERT.

5. Retrain the deployed models with the latest tweet data to promote progressive accuracy improvement and advancements.