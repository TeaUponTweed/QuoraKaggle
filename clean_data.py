import functools
import operator
import re

import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm

tqdm.pandas()
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

def build_vocab(sentences, verbose =  True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab




def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:
            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium',
                'Brexit': 'Britain exit'
                }

mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


def get_cleaned_sentences(df):
    sentences = df["question_text"].progress_apply(lambda x: clean_numbers(x)) \
                                   .progress_apply(lambda x: clean_text(x)) \
                                   .progress_apply(lambda x: replace_typical_misspell(x))

    sentences = sentences.progress_apply(lambda x: x.split())
    to_remove = ['a','to','of','and']
    sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
    return sentences



class SpellChecker(object):
    def __init__(self, words):
        self.words = words

    def P(self, word, N=None):
        N = N if N is not None else sum(self.words.values())
        "Probability of `word`."
        return self.words[word] / N


    def correction(self, word):
        if word in self.words:
            return word
        "Most probable spelling correction for word."
        try:
            newword = max(self.candidates(word), key=self.P)
        except KeyError:
            newword = word
        return newword


    def candidates(self, word): 
        "Generate possible spelling corrections for word."
        return self.known((word,)) or self.known(self.edits1(word)) or (word,)


    def known(self, words): 
        "The subset of `words` that appear in the dictionary of WORDS."
        return frozenset(w for w in words if w in self.words)


    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)


    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))


def main():
    news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
    embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)

    train_sentences = get_cleaned_sentences(train)
    test_sentences = get_cleaned_sentences(test)
    vocab = build_vocab(train_sentences + test_sentences)

    WORDS = {word: count for word, count in vocab.items() if word in embeddings_index}
    spell_checker = SpellChecker(WORDS)

    @functools.lru_cache(None)
    def correct_spelling(word):
        return spell_checker.correction(word)

    def gen_spellchecked_sentences(sentences):
        for sentence in tqdm(sentences):
            yield list(map(correct_spelling, sentence ))

    new_train_sentences = list(gen_spellchecked_sentences(train_sentences))
    new_test_sentences = list(gen_spellchecked_sentences(test_sentences))

    vocab = build_vocab(new_train_sentences + new_test_sentences)
    oov = check_coverage(vocab,embeddings_index)


if __name__ == '__main__':
    main()
