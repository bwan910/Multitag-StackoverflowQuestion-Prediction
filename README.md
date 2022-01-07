# Multitag-StackoverflowQuestion-Prediction: Project Overview

- Created a tool that predicts tags for Stackoverflow questions. 
- Exploratory Data Analysis were performed to explore dataset
- Preproccessed the data 
- Optimized SGDClassifier, LogisticRegression and LinearSVC using ClassfierChain to reach the best model.
- Model evaluated with a few metrics (Accuracy, Hamming loss and etc)
- Built a web interface using Streamlit

## Resources used 
**Python Version:** 3.8.7

**Dataset used:** [link](https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/stackoverflow.csv)


**Libraries used in model training:** pandas, numpy, matplotlib, nltk, re, sklearn, ast, pickle
joblib, pickle.

**Libraries for web app:** streamlit, requests, joblib, pickle, os


## Exploratory Data Analysis(EDA)
The following process are performed during EDA for this model which include: 

- Obtaining the summary of the dataset
- Removing the row of data which has the least frequent occur classes.
- Check if there are NULL values in the dataset.
- Check if there are any duplicated data.
- Plot a graph to visualise the number of each value for the Tags.

## Data Preprocessing 
The following process are performed during data preprocessing for this model which include: 
- Remove digits. 
- Remove stopwords. 

## Model Building

Firstly, apply MultiLabelBinarizer for the tags as labelencoding. Next apply TFIDF for the texts.  
I also split the data into train and tests sets with a test size of 20%.
Applied Classifeir chain method for multi-label classification problem transformation.

Tried three different models:
1) SGDClassifier
2) LogisticRegression
3) LinearSVC

## Model Performance
The LinearSVC model outperformed the other approaches based on the metrics below:
| Algorithms                                      | Jaccard Score | Hamming Loss | Accuracy |
|-------------------------------------------------|---------------|--------------|----------|
| Logistic Regression                             | 0.58          | 0.43         | 54%      |
| Stochastic Gradient Descent(SGD)                | 0.60          | 0.40         | 57%      |
| Linear Support Vector Classfication (LinearSVC) | 0.72          | 0.28         | 68%      |

## Productionization
In this phase, I built a web app interface with Streamlit. The web app takes in question texts and returns recommended tag. 
