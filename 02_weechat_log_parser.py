import time
import sys
import re
import nltk

from nltk.corpus import stopwords
stopset = list(set(stopwords.words('english')))

date_re = r"[0-9]{4}-[0-9]{2}-[0-9]{2}\s[0-9]{2}:[0-9]{2}:[0-9]{2}"
pseudo_re = r"\w+"
re_raw = r"(?P<date>"+date_re+")\s+\+*@*(?P<pseudo>"+pseudo_re+")\s+(?P<sentence>.*)"
re_compiled = re.compile(re_raw)

corpus = open('ps1_corpus_shuf.txt', 'r')
a=0
corpus_filtered = []

start = time.time()

for line in corpus:
    mat = re_compiled.match(line)
    if mat is not None:
        #print(mat.group('pseudo'), mat.group('sentence'))
        pseudo = mat.group('pseudo').lower()
        if pseudo in ['foo', 'bar']:
            corpus_filtered.append([pseudo, mat.group('sentence')])

print("Treated raw file in {}s".format(time.time()-start))
feat = time.time()

def word_features(post):
    features = []
    for word in nltk.word_tokenize(post):
        if word.lower() not in stopset:
            features.append([word.lower(), True])
    return dict(features)
     
corpus_filtered_limited = corpus_filtered[0:50000]
featuresets = [(word_features(key[1]), key[0])
               for key in corpus_filtered_limited]
size = int(len(featuresets) * 0.1)
print("Train set size: {}".format(size))
train_set, test_set = featuresets[size:],featuresets[:size]
print("Extracted features in {}s".format(time.time()-feat))
classif=time.time()
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("Classified in {}s".format(time.time()-classif))
print(nltk.classify.accuracy(classifier, test_set))
