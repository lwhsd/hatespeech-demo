from transformers import AutoTokenizer, BertTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION', )
from subprocess import call
# call(["nvcc", "--version"]) does not work
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
# call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('Active CUDA Device: GPU', torch.cuda.current_device())
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())
print ('Device name ', torch.cuda.get_device_name(0))

BERT = 'BERT'
TFIDF = 'TFIDF'
IDB_INDOBENCH_LITE = "indobenchmark/indobert-lite-base-p1"
IDB_INDOBENCH_BASE = "indobenchmark/indobert-base-p1"
IDB_INDOBENCH_LARGE = "indobenchmark/indobert-large-p1"


class EmbeddingModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def embed(self, train, test=None):
        if self.model_name == BERT:
            return self.BERT_embedding(train)

        if self.model_name == TFIDF:
            return self.tfidf_vecotrizer(train, test)

    def encode_tokenizer(self, text, tokenizer):
        tokendata = tokenizer.encode(
            text, truncation=True, max_length=512, add_special_tokens=True)
        return tokendata

    def BERT_embedding(self, task):

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"  

        device = torch.device(dev)
        tokenizer = BertTokenizer.from_pretrained(
            IDB_INDOBENCH_LITE)

        bert_model = AutoModel.from_pretrained(IDB_INDOBENCH_LITE)
        encoded_token = task['text'].apply(
            self.encode_tokenizer, tokenizer=tokenizer)
        tokens = [x for x in encoded_token]
        max_len = max([len(i) for i in tokens])

        padded = np.array([i + [0]*(max_len-len(i)) for i in tokens])
        attention_mask = np.where(padded != 0, 1, 0)

        input_ids = torch.tensor(padded).to(device)

        attention_mask = torch.tensor(attention_mask).to(device)
        with torch.no_grad():
            gpu_model = bert_model.to(device)
            last_hidden_states = gpu_model(
                input_ids, attention_mask=attention_mask)
        
        #last_hidden_states = last_hidden_states.cpu()
        features = last_hidden_states[0][:, 0, :]
        features.cpu().numpy()
        return features

    def tfidf_vecotrizer(self, train, test):
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 2))

        tfidf_training_features = tfidf_vectorizer.fit_transform(
            train['text'])
        tfidf_test_features = tfidf_vectorizer.transform(test['text'])

        return tfidf_training_features, tfidf_test_features
