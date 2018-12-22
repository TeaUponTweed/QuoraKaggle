import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics, linear_model, multioutput
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
tfidf = {}
tfidf['train'] = tfidf_vec.transform(train['question_text'].values.tolist())
tfidf['test'] = tfidf_vec.transform(test['question_text'].values.tolist())

# Format
out_data = {'train': train['target'].values, 'test': test['target'].values}

class Predictor(object):
    def __init__(self, model, kf_splits, out_data, in_data):
        self.model = model
        self.kf = KFold(n_splits=kf_splits, shuffle=True, random_state=SEED)
        self.out_data = out_data
        self.in_data = in_data
        if callable(getattr(self.model, 'predict_proba', None)):
            self.predict = lambda x: self.model.predict_proba(x)[:,1]
        else:
            self.predict = lambda x: self.model.predict(x)

    def fit(self):
        """ Fit model """
        print('Building model...')
        X, y = self.in_data['train'], self.out_data['train']
        X_test, y_test = self.in_data['test'], self.out_data['test']

        for n, (ix_dev, ix_val) in enumerate(self.kf.split(self.out_data['train'])):
            print(f'Fitting on split {n}')

            # Run model on development and validation datasets
            dev_X, val_X = X[ix_dev], X[ix_val]  # X values
            dev_y, val_y = y[ix_dev], y[ix_val]  # y values
            self.model.fit(dev_X, dev_y)

            # Predict and validate
            pred_train = self.predict(val_X)
            train_score = metrics.log_loss(val_y, pred_train)
            print(f'Train: {train_score}')
            pred_test = self.predict(X_test)
            test_score = metrics.log_loss(y_test, pred_test)
            print(f'Test: {test_score}')

class Ensembler(object):
    def __init__(self, predictors, model):
        self.model = multioutput.MultiOutputRegressor(model)
        self.kf = predictors[0].kf
        self.in_data = predictors[0].in_data
        self.out_data = predictors[0].out_data
        self.out_weights = {}
        self.predictors = predictors

        # Format output as weights of the various predictors
        for t in ['train','test']:
            truth = predictors[0].out_data[t][:]
            error = np.zeros([truth.shape[0], len(predictors)])
            for ixp, p in enumerate(predictors):
                error[:,ixp] = np.abs(truth - p.predict(self.in_data[t]))
            
            self.out_weights[t] = np.zeros(error.shape)
            for ix_row in range(error.shape[0]):
                inv_err = 1 / error[ix_row,:] 
                self.out_weights[t][ix_row,:] = inv_err / np.sum(inv_err)

        # Set prediction function
        if callable(getattr(self.model, 'predict_proba', None)):
            self.f_predict = lambda x: self.model.predict_proba(x)[:,1]
        else:
            self.f_predict = lambda x: self.model.predict(x)

    def predict(self, x):
        weights = self.model.predict(x)
        preds = np.array([p.predict(x) for p in self.predictors]).T
        return np.sum(weights*preds, 1)

    def fit(self):
        """ Fit model """
        print('Building model...')
        X, w, y = self.in_data['train'], self.out_weights['train'], self.out_data['train']
        X_test, y_test = self.in_data['test'], self.out_data['test']

        for n, (ix_dev, ix_val) in enumerate(self.kf.split(self.out_data['train'])):
            print(f'Fitting on split {n}')

            # Run model on development and validation datasets
            dev_X, val_X = X[ix_dev], X[ix_val]  # X values
            dev_w, val_w = w[ix_dev], w[ix_val]  # w values
            dev_y, val_y = y[ix_dev], y[ix_val]  # y values
            self.model.fit(dev_X, dev_w)

            # Predict and validate
            pred_train = self.predict(val_X)
            train_score = metrics.log_loss(val_y, pred_train)
            print(f'Train: {train_score}')
            pred_test = self.predict(X_test)
            test_score = metrics.log_loss(y_test, pred_test)
            print(f'Test: {test_score}')

# Logistic Regression
logr_reg_model = linear_model.LogisticRegression(C=5., solver='sag')
pred_logr_reg = Predictor(logr_reg_model, 3, out_data, tfidf)
pred_logr_reg.fit()

# Ridge Regression
rid_reg_model = linear_model.Ridge()
pred_rid_reg = Predictor(rid_reg_model, 3, out_data, tfidf)
pred_rid_reg.fit()

# Ensemble
predictors = [pred_logr_reg, pred_rid_reg]
ens_model = linear_model.Ridge()
ens = Ensembler(predictors, ens_model)
ens.fit()

# Check out results
pred = ens.predict(tfidf['test'])
test_y = out_data['test']
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