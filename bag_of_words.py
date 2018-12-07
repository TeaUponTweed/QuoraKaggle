import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold

SEED = 12378

# Load data
all_data = pd.read_csv('../input/train.csv')
train, test = train_test_split(all_data, test_size=0.1, random_state=SEED)

# Get the tfidf vectors
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))

# Tokenize some punctuation
for punc in '?!-*=+':
    tfidf_vec.token_pattern += '|\\' + punc

# Fit everything to the model
tfidf_vec.fit_transform(all_data['question_text'].values.tolist())
train_tfidf = tfidf_vec.transform(train['question_text'].values.tolist())
test_tfidf = tfidf_vec.transform(test['question_text'].values.tolist())

def runModel(train_X, train_y, test_X, test_y, test_X2):
    model = linear_model.LogisticRegression(C=5., solver='sag')
    model.fit(train_X, train_y)  # Fit model on new training data
    pred_test_y = model.predict_proba(test_X)[:,1]  # Predict on training data
    pred_test_y2 = model.predict_proba(test_X2)[:,1]  # Predict on test data
    return pred_test_y, pred_test_y2, model

# Train model
print("Building model.")
kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
train_y = train['target'].values

pred_train = np.zeros([train.shape[0]])
pred_test, cv_scores = [], []
for n, (dev_index, val_index) in enumerate(kf.split(train)):
    print(f'Fitting on split {n}')
    
    # Run model on development and validation datasets
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runModel(dev_X, dev_y, val_X, val_y, test_tfidf)
    
    # Update test and training predictions
    assert all([v==0 for v in pred_train[val_index]])
    pred_test.append(pred_test_y)
    pred_train[val_index] = pred_val_y
    
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))


# Check out results
plt.figure()
plt.plot(cv_scores)
plt.plot([metrics.log_loss(test['target'].values, v) for v in pred_test])

pred = model.predict_proba(test_tfidf)[:,1]
test_y = test['target'].values
scores = []
thresholds = np.arange(0, 1, 0.01)
for thresh in thresholds:
    thresh = np.round(thresh, 2)
    score = metrics.f1_score(test_y, (pred>thresh).astype(int))
    scores.append(score)
    
plt.figure()
plt.plot(thresholds, scores)

best_score = max(scores)
best_thresh = thresholds[np.argmax(scores)]
print(f"Best score is {best_score} at threshold {best_thresh}")

plt.show()