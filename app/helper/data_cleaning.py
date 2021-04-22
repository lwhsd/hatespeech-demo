from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory,StopWordRemover,ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np 
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer 
nltk.download('punkt')
from nltk.corpus import stopwords 
nltk.download('stopwords')
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

ADD_STOPWORDS =["(",")","senin","selasa","rabu","kamis","jumat","sabtu","minggu"]

class DataCleaning:
    def __init__(self, text):
        self.text = text
    
    def remove_numbers(self):
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        words= tokenizer.tokenize(self.text)
        return " ".join(words)

    def remove_punctuation(self):
        words = self.text.split()
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in words]
        return " ".join(stripped)

    def stem_text(self):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return stemmer.stem(self.text)   

    def remove_stopwords(self):
        default_stopwords = StopWordRemoverFactory().get_stop_words()
        dictionary=ArrayDictionary(default_stopwords+ADD_STOPWORDS)
        id_stopword = StopWordRemover(dictionary)
        return id_stopword.remove(self.text)

    def remove_english_stopwords(self):
        en_stopword = set(stopwords.words('english'))
        if self.text:
            return " ".join([token for token in self.text.split() if token not in en_stopword])

    def stem_english_text(self):
        en_stemmer = PorterStemmer()
        wnl = WordNetLemmatizer()
       
        return " ".join([ wnl.lemmatize(word) if wnl.lemmatize(word).endswith('e') else en_stemmer.stem(word) for word in self.text.split()])

    def remove_single_char(self):
        return " ".join([ x for x in self.text.split() if (len(x)>1 and isinstance(x, str))])

