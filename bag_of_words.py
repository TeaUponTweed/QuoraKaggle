import string

import os
import json
import string
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import eli5


from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

from sklearn.pipeline import Pipeline

from wordcloud import WordCloud, STOPWORDS

train_df = pd.read_csv("../input/train.csv")
# test_df = pd.read_csv("../input/test.csv")
# print("Train shape : ", train_df.shape)
# print("Test shape : ", test_df.shape)
# all_df = pd.read_csv("../input/train.csv")
# train_df, test_df = train_test_split(0.1, )
train_df, test_df = train_test_split(train_df, test_size=0.1, random_state=12378)


# Get the tfidf vectors #
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
tfidf_vec.fit_transform(train_df['question_text'].values.tolist() + test_df['question_text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['question_text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['question_text'].values.tolist())
train_y = train_df["target"].values

def runModel(train_X, train_y, test_X, test_y, test_X2):
    model = linear_model.LogisticRegression(C=5., solver='sag')
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)[:,1]
    pred_test_y2 = model.predict_proba(test_X2)[:,1]
    return pred_test_y, pred_test_y2, model

print("Building model.")
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0]])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_df):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runModel(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    break

def threshscore(pred, truth):
    for thresh in np.arange(0.01, 0.801, 0.01):
        thresh = np.round(thresh, 2)
        print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(truth, (pred>thresh).astype(int))))

class PuncClassifier(object):
    def __init__(self, symbols):
        self.symbols = symbols
        self.probs = {}
        self.max_count = 2
    def fit(self, X, y):
        for sym in self.symbols:
            punc_counts = []
            for sentence in X:
                punc_counts.append( sentence.count(sym) )
            self.probs[sym] = np.zeros(self.max_count+1)
            for n in range(self.max_count):
                res = [t for v, t in zip(punc_counts, y) if v == n]
                if len(res) == 0:
                    self.probs[sym][n] = 0
                else:
                    self.probs[sym][n] = np.mean(res)
            n = self.max_count
            res = [t for v, t in zip(punc_counts, y) if v >= n]
            if len(res) == 0:
                self.probs[sym][n] = 0
            else:
                self.probs[sym][n] = np.mean(res)
            print(sym, self.probs[sym])

    def predict(self, X):
        out = []
        for sentence in X:
            p_sincere = 1.0
            for sym in self.symbols:
                p_insincere = self.probs[sym][min(sentence.count(sym), self.max_count)]
                p_sincere = (1 - p_insincere) * p_sincere
                
            out.append(1 - p_sincere)

        return np.array(out)

pc = PuncClassifier('?!-*')
pc.fit(train_df['question_text'].values, train_df['target'].values)

pred = pc.predict(test_df['question_text'])

wakka = 1 - (model.predict_proba(test_tfidf)[:, 0] * (1 - pred))

threshscore(wakka, test_df['target'])
