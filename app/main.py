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
#task = task.head(1350)
train, test = train_test_split(
    task, test_size=0.2, random_state=42, shuffle=True)

start_time = time.time()
print (train.shape)
#train = train.head(500)
#train_1 = train.tail(896)
#test_0 = train.head(224)
#test_1 = train.tail(224)
print (train.shape)
print (test.shape)
_0, _1, _2, _3 = np.array_split(train, 4)
train_0 = EmbeddingModel('BERT').embed(_0)
train_1 = EmbeddingModel('BERT').embed(_1)
train_2 = EmbeddingModel('BERT').embed(_2)
train_3 = EmbeddingModel('BERT').embed(_3)
features_train = np.concatenate((train_0.cpu(), train_1.cpu(), train_2.cpu(), train_3.cpu()), axis=0)
#features_train_0 = EmbeddingModel('BERT').embed(train)
#features_train_1 = EmbeddingModel('BERT').embed(train_1)
#features_train = EmbeddingModel('BERT').embed(train)
#features_train = np.concatenate((features_train_0, features_train_1), axis=0)
features_test = EmbeddingModel('BERT').embed(test)
# print(features)
# features_train, features_test = EmbeddingModel('TFIDF').embed(train, test)
# print(features_train)
# print(features_test)
classifiers = Classifier("SVM").classify(features_train, train["label"])
Evaluation().evaluate(classifiers, features_test.cpu(), test["label"])
print("Running model --- %s seconds ---" % (time.time() - start_time))
