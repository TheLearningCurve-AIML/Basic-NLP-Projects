import os
from google.cloud import storage
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction .text import CountVectorizer
from fuzzywuzzy import fuzz
from nltk.stem import PorterStemmer


def CommonWords(row):
  w1 = set(map(lambda word: word.lower().strip(), row["question1"].split(" ")))
  w2 = set(map(lambda word: word.lower().strip(), row["question2"].split(" ")))
  return len(w1 & w2)

def total_words(row):
  w1 = set(map(lambda word: word.lower().strip(), row["question1"].split(" ")))
  w2 = set(map(lambda word: word.lower().strip(), row["question2"].split(" ")))
  return (len(w1) + len(w2))

def fetch_token_features(row):
    q1 = row['question1']
    q2 = row['question2']
    SAFE_DIV = 0.0001

    # Manual Stopwords to avoid internet download error
    STOP_WORDS = {"a", "an", "the", "and", "or", "but", "if", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing"}

    token_features = [0.0]*8
    q1_tokens = str(q1).split()
    q2_tokens = str(q2).split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features

def apply_basic_feature(data):
    # Lower the character
    data['q1_len'] = data['question1'].str.len()
    median = data["q1_len"].median()
    iqr = data["q1_len"].quantile(0.75)- data["q1_len"].quantile(0.25)
    data['q1_len'] = (data["q1_len"]-median)/iqr

    data['q2_len'] = data['question2'].str.len()
    median = data["q2_len"].median()
    iqr = data["q2_len"].quantile(0.75)- data["q2_len"].quantile(0.25)
    data['q2_len'] = (data["q2_len"]-median)/iqr

    # Question Length
    data["q1_num_words"] = data["question1"].apply(lambda row: len(row.split(" ")))
    median = data["q1_num_words"].median()
    iqr = data["q1_num_words"].quantile(0.75) - data["q1_num_words"].quantile(0.25)
    data['q1_num_words'] =  (data["q1_num_words"]-median)/iqr


    data["q2_num_words"] = data["question2"].apply(lambda row: len(row.split(" ")))
    median = data["q2_num_words"].median()
    iqr = data["q2_num_words"].quantile(0.75) - data["q2_num_words"].quantile(0.25)
    data['q2_num_words'] =  (data["q2_num_words"]-median)/iqr



    # Number of common words
    data['word_common']= data.apply(CommonWords,axis=1)
    median = data['word_common'].median()
    iqr = data['word_common'].quantile(0.75) - data['word_common'].quantile(0.25)
    data['word_common'] =  (data['word_common']-median)/iqr

    # Total number of words
    data['total_words']= data.apply(total_words,axis=1)
    median = data['total_words'].median()
    iqr = data['total_words'].quantile(0.75) - data['total_words'].quantile(0.25)
    data['total_words'] =  (data['total_words']-median)/iqr
    


    # Word share
    data['word_share']= round(data['word_common']/data['total_words'],2)

    return data

def apply_advance_feature(data):
    data['fuzz_ratio'] = data.apply(lambda row: fuzz.ratio(row['question1'], row['question2']), axis=1)/100
    data['fuzz_partial_ratio'] = data.apply(lambda row: fuzz.partial_ratio(row['question1'], row['question2']), axis=1)/100
    data['fuzz_token_sort_ratio'] = data.apply(lambda row: fuzz.token_sort_ratio(row['question1'], row['question2']), axis=1)/100
    data['fuzz_token_set_ratio'] = data.apply(lambda row: fuzz.token_set_ratio(row['question1'], row['question2']), axis=1)/100

    token_features = data.apply(fetch_token_features, axis=1)

    data["cwc_min"]       = list(map(lambda x: x[0], token_features))
    data["cwc_max"]       = list(map(lambda x: x[1], token_features))
    data["csc_min"]       = list(map(lambda x: x[2], token_features))
    data["csc_max"]       = list(map(lambda x: x[3], token_features))
    data["ctc_min"]       = list(map(lambda x: x[4], token_features))
    data["ctc_max"]       = list(map(lambda x: x[5], token_features))
    data["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    data["first_word_eq"] = list(map(lambda x: x[7], token_features))

    return data



def predict(input_data):
    # print("In predict function")
    data = pd.DataFrame(input_data, columns=['question1', 'question2'])
    vectorizer = joblib.load("vectorizer.joblib")
    ques1_arr = vectorizer.transform(data['question1']).toarray()
    temp_df1 = pd.DataFrame(ques1_arr)
    temp_df1.index = data.index

    quest2_arr = vectorizer.transform(data['question2']).toarray()
    temp_df2 = pd.DataFrame(quest2_arr)
    temp_df2.index = data.index

    data = pd.concat([data, temp_df1, temp_df2], axis=1)
    data = apply_basic_feature(data)
    data = apply_advance_feature(data)
    data = data.iloc[:, 2:]
    data.columns = data.columns.astype(str)
    
    model = joblib.load("model.joblib")
    prediction = model.predict(data.values)

    return prediction



    