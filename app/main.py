from helper.data_cleaning import DataCleaning
from helper.embedding_model import EmbeddingModel
from helper.classifier import Classifier, Evaluation
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time


def clean_text(file_path="app/temp/indo_hs.txt"):
    print(file_path)
    task = pd.read_csv(file_path, sep=",", header=0, error_bad_lines=False)
    task = task[task.text.notna()]
    task.text = task.text.str.lower()
    task.text = task.text.apply(lambda x: DataCleaning(x).remove_numbers())
    task.loc[(task['lang'] == 'id'),'text'] = task.loc[(task['lang'] == 'id'),'text'].apply(lambda x: DataCleaning(x).remove_stopwords())
    task.loc[(task['lang'] == 'id'),'text'] = task.loc[(task['lang'] == 'id'),'text'].apply(lambda x: DataCleaning(x).stem_text())
    task.loc[(task['lang'] == 'en'),'text'] = task.loc[(task['lang'] == 'en'),'text'].apply(lambda x: DataCleaning(x).remove_english_stopwords())
    task.loc[(task['lang'] == 'en'),'text'] = task.loc[(task['lang'] == 'en'),'text'].apply(lambda x: DataCleaning(x).stem_english_text())
    task.text = task.text.apply(lambda x: DataCleaning(x).remove_single_char())
    task = task[task.text.notna()]
    task.to_csv('app/temp/mix_clean.csv')

    print(task.head(10))



# clean_text('app/temp/mix_hs.csv')
task = pd.read_csv("app/temp/mix_clean.csv")
train, test = train_test_split(
    task, test_size=0.2, random_state=42, shuffle=True)

start_time = time.time()
# features_train = EmbeddingModel('BERT').embed(train)
# features_test = EmbeddingModel('BERT').embed(test)
# print(features)
features_train, features_test = EmbeddingModel('TFIDF').embed(train, test)
# print(features_train)
# print(features_test)
classifiers = Classifier("SVM").classify(features_train, train["label"])
Evaluation().evaluate(classifiers, features_test, test["label"])
print("Running model --- %s seconds ---" % (time.time() - start_time))
