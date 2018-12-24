import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics, linear_model, multioutput
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from gensim.models import KeyedVectors

SEED = 12378

def load_data():
    # Load raw data
    print('Loading raw data...')
    raw_data = pd.read_csv('../input/train.csv')
    raw = {}
    raw['train'], raw['test'] = train_test_split(raw_data, test_size=0.1, random_state=SEED)
    
    # Load cleaned data
    print('Loading cleaned data...')
    clean_data = pd.read_csv('../input/clean_train.csv')
    clean = {}
    clean['train'], clean['test'] = train_test_split(clean_data, test_size=0.1, random_state=SEED)
    
    assert all(np.equal(raw['train'].qid, clean['train'].qid)), 'Train data doesn\'t match'
    assert all(np.equal(raw['test'].qid, clean['test'].qid)), 'Test data doesn\'t match'

    # Load embeddings
    print('Loading word2vec embeddings...')
    emb_file = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
    emb_model = KeyedVectors.load_word2vec_format(emb_file, binary=True)
    print('Converting to embeddings...')
    emb_bags = {'train': word2vec_bags(clean['train'], emb_model), 
                'test': word2vec_bags(clean['test'], emb_model)}

    # Create tfidfs
    print('Building raw tfidfs...')
    raw_tfidf = fit_tfidf(raw_data, raw['train'], raw['test'])
    print('Building clean tfidfs...')
    clean_tfidf = fit_tfidf(clean_data, clean['train'], clean['test'])

    def build_df(type_):
        return {
            'raw_text':     raw[type_].question_text.values, 
            'raw_tfidf':    raw_tfidf[type_], 
            'clean_text':   clean[type_].question_text.values, 
            'clean_tfidf':  clean_tfidf[type_], 
            'emb_bags':     emb_bags[type_],
            'target':       raw[type_].target.values,
            'qid':          raw[type_].qid.values }

    # Combine
    return {'train': build_df('train'), 'test': build_df('test')}

def fit_tfidf(all_data, train, test):
    """ Get the tfidf vectors for inputted data """
    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))

    # Tokenize some punctuation
    for punc in '?!-*=+':
        tfidf_vec.token_pattern += '|\\' + punc

    # Fit everything to the model
    tfidf_vec.fit_transform(all_data['question_text'].values.tolist())
    tfidf = {}
    tfidf['train'] = tfidf_vec.transform(train['question_text'].values.tolist())
    tfidf['test'] = tfidf_vec.transform(test['question_text'].values.tolist())
    return tfidf

def word2vec_bags(data, emb_model):
    """ For each sentence, add the constituent vectors """
    sentences = data['question_text'].values.tolist()
    sums = []
    for s in sentences:
        vecs = [emb_model.get_vector(w) for w in s.split(' ') if w in emb_model]
        sums.append( np.sum(vecs, axis=0) )
    return np.array(sums)

class Predictor(object):
    def __init__(self, model, kf_splits, data, data_type):
        self.model = model
        self.kf = KFold(n_splits=kf_splits, shuffle=True, random_state=SEED)
        self.data = {'train': {'X': data['train'][data_type], 'y': data['train']['target']}, 
                     'test':  {'X': data['test'][data_type],  'y': data['test']['target']}}
        if callable(getattr(self.model, 'predict_proba', None)):
            self.predict = lambda x: self.model.predict_proba(x)[:,1]
        else:
            self.predict = lambda x: self.model.predict(x)

    def fit(self):
        """ Fit model """
        print('Building model...')

        X, y = self.data['train']['X'], self.data['train']['y']
        for n, (ix_dev, ix_val) in enumerate(self.kf.split(y)):
            print(f'Fitting on split {n}')

            # Run model on development and validation datasets
            dev_X, val_X = X[ix_dev], X[ix_val]  # X values
            dev_y, val_y = y[ix_dev], y[ix_val]  # y values
            self.model.fit(dev_X, dev_y)

            # Predict and validate
            pred_train = self.predict(val_X)
            train_score = metrics.log_loss(val_y, pred_train)
            print(f'Train: {train_score}')

            pred_test = self.predict(self.data['test']['X'])
            test_score = metrics.log_loss(self.data['test']['y'], pred_test)
            print(f'Test: {test_score}')

class Ensembler(object):
    def __init__(self, predictors, model, kf_splits, data, data_type):
        self.model = multioutput.MultiOutputRegressor(model)
        self.predictors = predictors
        self.kf = KFold(n_splits=kf_splits, shuffle=True, random_state=SEED)
        self.data = {'train': {'X': data['train'][data_type], 'y': data['train']['target'], 'w': {}}, 
                     'test':  {'X': data['test'][data_type],  'y': data['test']['target'], 'w': {}}}

        # Format output as weights of the various predictors
        for t in self.data.keys():
            truth = self.data[t]['y'][:]
            error = np.zeros([truth.shape[0], len(predictors)])
            for ixp, p in enumerate(predictors):
                error[:,ixp] = np.abs(truth - p.predict(p.data[t]['X']))
            
            self.data[t]['w'] = np.zeros(error.shape)
            for ix_row in range(error.shape[0]):
                inv_err = 1 / error[ix_row,:] 
                self.data[t]['w'][ix_row,:] = inv_err / np.sum(inv_err)

    def predict(self, ens_x, pred_x):
        weights = self.model.predict(ens_x)
        preds = np.array([p.predict(x) for x,p in zip(pred_x, self.predictors)]).T
        return np.sum(weights*preds, 1)

    def fit(self):
        """ Fit model """
        print('Building model...')

        X, w, y = self.data['train']['X'], self.data['train']['w'], self.data['train']['y']
        for n, (ix_dev, ix_val) in enumerate(self.kf.split(y)):
            print(f'Fitting on split {n}')

            # Run model on development and validation datasets
            dev_X, val_X = X[ix_dev], X[ix_val]  # X values
            dev_w, val_w = w[ix_dev], w[ix_val]  # w values
            dev_y, val_y = y[ix_dev], y[ix_val]  # y values
            self.model.fit(dev_X, dev_w)

            # Predict and validate
            p_val_X = [p.data['train']['X'][ix_val] for p in self.predictors]
            pred_train = self.predict(val_X, p_val_X)
            train_score = metrics.log_loss(val_y, pred_train)
            print(f'Train: {train_score}')

            p_test_X = [p.data['test']['X'] for p in self.predictors]
            pred_test = self.predict(self.data['test']['X'], p_test_X)
            test_score = metrics.log_loss(self.data['test']['y'], pred_test)
            print(f'Test: {test_score}')


def plot_results(data, pred, do_plot=False):
    scores = []
    thresholds = np.arange(0, 1, 0.001)
    for thresh in thresholds:
        thresh = np.round(thresh, 2)
        score = metrics.f1_score(data['test']['target'], (pred>thresh).astype(int))
        scores.append(score)

    best_score = max(scores)
    best_thresh = thresholds[np.argmax(scores)]
    print(f"Best score is {best_score} at threshold {best_thresh}")

    if do_plot:
        plt.figure()
        plt.plot(thresholds, scores)
        plt.show()


def main():
    data = load_data()

    # Logistic Regression
    model = linear_model.LogisticRegression(C=5., solver='sag')
    p_logistic_raw = Predictor(model, 3, data, 'raw_tfidf')
    p_logistic_raw.fit()
    p_logistic_clean = Predictor(model, 3, data, 'clean_tfidf')
    p_logistic_clean.fit()

    # Ridge Regression
    model = linear_model.Ridge()
    p_ridge_raw = Predictor(model, 3, data, 'raw_tfidf')
    p_ridge_raw.fit()
    p_ridge_clean = Predictor(model, 3, data, 'clean_tfidf')
    p_ridge_clean.fit()

    # Ensembler
    predictors = [p_logistic_raw, p_logistic_clean, p_ridge_raw, p_ridge_clean]
    model = linear_model.Ridge()
    ens = Ensembler(predictors, model, 3, data, 'raw_tfidf')
    ens.fit()

    # Show results
    pred = ens.predict(ens.data['test']['X'], [p.data['test']['X'] for p in ens.predictors])
    plot_results(data, pred)

if __name__ == '__main__':
    main()