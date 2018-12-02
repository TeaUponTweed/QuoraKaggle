import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

from sklearn.pipeline import Pipeline

train_df = pd.read_csv("../input/train.csv")
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=12378)



from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('reg', LogisticRegression())
])

text_clf.fit(train_df['question_text'], train_df['target'])

predicted = text_clf.predict(val_df['question_text'])
for thresh in [0.5]:
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, f1_score(val_df['target'], (predicted>thresh).astype(int))))

print("Baseline F1 score: ", f1_score(val_df['target'], np.ones(val_df['target'].size).astype(int)))