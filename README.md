# Project Summary: Tweet Sexism Detection and Classification

This project is part of the **UPV ETSINF Natural Language and Information retrieval classes** and focuses on detecting sexism in tweets and classifying it into various categories. The project is designed to explore different machine learning models and representations of text data to address issues related to sexism on social media platforms. The ultimate goal is to participate in **EXIST 2025**, a shared task on sexism detection.

### Notebooks Overview

1. **Preprocessing and Traditional Language Representation for Text Classification using Scikit-Learn**  
   This notebook delves into the traditional techniques for text classification using **Scikit-learn**, including **TF-IDF** and **CountVectorizer**. The focus is on preprocessing text data to prepare it for machine learning models, while examining the performance of various traditional representations in comparison to newer word embedding approaches.


2. **Static Word Embeddings for Text Representation**  
   This notebook introduces **static word embeddings**, such as **Word2Vec** and **GloVe**, to represent words in a fixed, continuous vector space. The goal is to explore how these word embeddings can be used for tweet classification and whether they provide better results than traditional methods.


3. **Contextual Word Embeddings for Text Representation**  
   In this notebook, we explore **contextual word embeddings**, focusing on models like **BERT** and **RoBERTa**. These embeddings provide a more dynamic and context-dependent representation of words, which can improve performance in tasks like sexism detection. The notebook also evaluates how these models perform in the task of identifying sexism in tweets compared to static word embeddings.


4. **Sexism Detection (Subtask 1)**  
   This notebook focuses on detecting whether a tweet contains sexist content or not. It uses multiple machine learning models, including **Logistic Regression**, **Decision Trees**, and **MLPs**. The text data is represented using two techniques:
   - **LSA-based TF-IDF (50 components)**
   - **RoBERTa contextual embeddings**  
   This task is aimed at training and evaluating models to classify tweets as sexist or non-sexist.


5. **Sexism Classification (Subtask 2)**  
   After detecting sexism in tweets, this notebook classifies the sexist tweets into four categories:
   - **JUDGEMENTAL**
   - **REPORTED**
   - **DIRECT**
   - **UNKNOWN**  
   This task involves fine-tuning the classifiers to improve the accuracy of the classification process, focusing on handling multi-class classification.

### Techniques and Tools Used
- **Text Representation**: TF-IDF, LSA, Word2Vec, GloVe, BERT, RoBERTa embeddings
- **Classifiers**: Logistic Regression, MLP (Multilayer Perceptron), Decision Tree
- **Libraries**: scikit-learn, pandas, numpy, transformers (for RoBERTa)

### Goal
The project aims to build an efficient pipeline for sexism detection and classification in tweets, focusing on understanding different approaches to NLP tasks and applying machine learning models effectively in the context of social media data. The goal is to participate in **EXIST 2025**, a series of scientific events and shared tasks on sexism identification in social networks, hosted as part of CLEF 2025 in Madrid, Spain, from September 9-12, 2025. EXIST fosters the automatic detection of sexism in a broad sense, from explicit misogyny to subtle expressions involving implicit sexist behaviors.