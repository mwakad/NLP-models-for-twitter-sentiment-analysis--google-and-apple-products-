# Twitter Sentiment Analysis for Google and Apple Products
## Project Summary
This project focuses on performing sentiment analysis on tweets related to Apple and Google products. Leveraging a dataset of over 9,000 human-rated tweets, the goal is to classify sentiments as positive, negative, or neutral to understand public perception and identify trends. The project aims to build and evaluate multiple models for predicting sentiment (**_Multinomial Naive Bayes, K-Nearest Neighbor, Random Forest Classifier, and Neural Networks_**, and a **_Logistic Regression Model_** for predicting the product an emotion is directed at. The best-performing **_sentiment prediction model_** and the **_Logistic Regression_** model are integrated into a Streamlit deployment application for real-time insights.

## Data Understanding
The dataset used is tweet_product_company.csv, which contains real-world tweet data mentioning Apple and Google products. Each entry includes the tweet text, the referenced product (e.g., iPhone, Google), the associated company (Apple or Google), and a sentiment label (positive, negative, or neutral). The dataset is valuable due to the short, noisy, and opinion-driven nature of tweets, making it ideal for testing robust NLP techniques. It supports both binary and multiclass sentiment prediction and offers real-world variability, including emoji usage, slang, and abbreviations.

## Problem Statement
In today's digital age, social media serves as a vast platform for customers to express their opinions. For major tech brands like Apple and Google, understanding these sentiments is crucial. Sentiment analysis provides deep insights into public perception, allowing businesses to gauge customer emotions and proactively address concerns, ultimately leading to improved products and services.

## Business Objectives
The primary client for this NLP project is Apple. By analyzing sentiments from tweets about their products and those of their competitors, Apple can gain authentic feedback that traditional methods might miss. This real-time access to customer sentiment will enable Apple to quickly identify trends, preferences, and potential issues, facilitating proactive engagement and timely adjustments to their strategies.

## Project Objectives:
### Objective 1: Sentiment Distribuction by Company 
EDA revealed the distribution of sentiments across Apple and Google products. After cleaning and categorizing products, the sentiment distribution showed that a significant portion of tweets were neutral, followed by positive, and then negative sentiments. This highlights the importance of distinguishing between no emotion and actual positive/negative feedback.

### Objective 2: Logistic Regression Model for Predicting Company 
A Logistic Regression model was build and tuned to classify tweets as pertaining to either Apple or Google products. The model serves as a preliminary classification step before sentiment prediction by predicting which company's products is an emotion sentiments in a tweet directed at.

**Logistic Regression Model**

|             | precision | recall | f1-score | support |
|-------------|-----------|--------|----------|---------|
| Apple       | 0.98      | 0.99   | 0.99     | 1085    |
| Google      | 0.97      | 0.97   | 0.97     | 543     |
| Accuracy    |       |     |    0.98  | 1628    |
| macro avg   | 0.98      | 0.98   | 0.98     | 1628    |
| weighted avg| 0.98      | 0.98   | 0.98     | 1628    |


### Objective 3: Build and Tune Multiple Models for Predicting Sentiment
Four models were build, and tuned via GridSearchCV(). The classification reports for the tuned versions of the models are captured in the tables below:

**Multinomial Naves Bayes Model**

|             | precision | recall | f1-score | support |
|-------------|-----------|--------|----------|---------|
| Negative    | 0.34      | 0.35   | 0.35     | 91      |
| Neutral     | 0.73      | 0.66   | 0.69     | 742     |
| Positive    | 0.54      | 0.61   | 0.57     | 469     |
| Accuracy    |       |     |    0.62  | 1302    |
| macro avg   | 0.53      | 0.54   | 0.54     | 1302    |
| weighted avg| 0.63      | 0.62   | 0.62     | 1302    |

**K Nearest Neighbour Model**

|             | precision | recall | f1-score | support |
|-------------|-----------|--------|----------|---------|
| Negative    | 0.40      | 0.37   | 0.39     | 91      |
| Neutral     | 0.65      | 0.80   | 0.72     | 742     |
| Positive    | 0.61      | 0.39   | 0.47     | 469     |
| Accuracy    |       |     |    0.62  | 1302    |
| macro avg   | 0.55      | 0.52   | 0.53     | 1302    |
| weighted avg| 0.62      | 0.62   | 0.61     | 1302    |

**Random Forests Model**

|             | precision | recall | f1-score | support |
|-------------|-----------|--------|----------|---------|
| Negative    | 0.22      | 0.37   | 0.28     | 91      |
| Neutral     | 0.72      | 0.68   | 0.64     | 742     |
| Positive    | 0.52      | 0.61   | 0.56     | 469     |
| Accuracy    |       |     |    0.57  | 1302    |
| macro avg   | 0.48      | 0.52   | 0.49     | 1302    |
| weighted avg| 0.61      | 0.57   | 0.59     | 1302    |

**Sequential Neural Network**

|             | precision | recall | f1-score | support |
|-------------|-----------|--------|----------|---------|
| Negative    | 0.51      | 0.35   | 0.42     | 91      |
| Neutral     | 0.70      | 0.75   | 0.72     | 742     |
| Positive    | 0.58      | 0.56   | 0.57     | 469     |
| Accuracy    |       |     |    0.65  | 1302    |
| macro avg   | 0.60      | 0.55   | 0.57     | 1302    |
| weighted avg| 0.64      | 0.65   | 0.65     | 1302    |


### Objective 4: Evaluate and Compare Models' Performance
The performance of each sentiment model was rigorously evaluated based on: Accuracy, Precision, Recall, F1-score, and Confusion Matrices on the test set. This comparative analysis helped determine the most effective model for deployment, ensuring it is robust and generalizable to new, unseen tweet data. 

**Tuned Models' Performance on Test Set**

| Model              | Test Accuracy | Test Precision | Test Recall | Test F1  |
|--------------------|---------------|----------------|-------------|----------|
| Tuned MultinomialNB| 0.656232      | 0.655698       | 0.662776    | 0.656232 |
| Tuned KNN          | 0.602126      | 0.637935       | 0.639978    | 0.602126 |
| Tuned RandomForest | 0.663244      | 0.689657       | 0.685504    | 0.663244 |
| Neural Network     | 0.664619      | 0.661348       | 0.660934    | 0.657428 |

The tuned Random Forests model achieved the highest scores across all the metrics on the test set.


However the **Random Forest model** (19) lags behind the **Multinomial Naves Bayes** (31), and the **Neural Network** (32) in predicting the minority class (Negative Sentiment). This limitation of the Ensemble model is likely due to class imbalance even though the **_scoring_** parameter was set to `f1-weighted` during hyperparameter tuning (to reinforce improvements towards accuracy in making predictions for all classes). 




### Objective 5: Deployment
A user-friendly Streamlit application was developed to demonstrate the utility of the selected model. This app will allow users to input tweets and receive predictions on both the associated product (Apple or Google) and the sentiment (positive, negative, or neutral), providing a practical tool for real-time sentiment analysis.

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