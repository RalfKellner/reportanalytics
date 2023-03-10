from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
import pandas as pd
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def sentence_preprocessor(sentence):
    return simple_preprocess(strip_tags(sentence), deacc=True)

def text_preprocessor(raw_text, idx_start=0, min_words=10, max_words=100, **kwargs):
    
    # identify unique sentences
    sentences = sent_detector.tokenize(raw_text.strip())
    # preprocess each sentence
    sentences_prep = [sentence_preprocessor(sentence) for sentence in sentences]
    # check if sentences should be used; very uncommon sentences in terms of length may not be representative
    sentences_use = [(min_words <= len(sentence_prep) <= max_words) for sentence_prep in sentences_prep]

    # joint each token list for saving in the data frame
    sentences_join = [','.join(sentence_prep) for sentence_prep in sentences_prep]
    
    # check is something odd happens during the first two steps
    assert len(sentences) == len(sentences_join), 'See if something odd occurs for text preparation, raw number of sentences is not equal to the number of preprocessed sentences.'
    
    # generate an index for the dataframe; if one iterates through many texts, it may be desired to build an index over all texts
    idx = list(range(idx_start, idx_start + len(sentences), 1))
    # generate a dataframe
    sentences_df = pd.DataFrame({'raw_text': sentences, 'prep_text': sentences_join, 'use_doc': sentences_use}, index = idx)
    
    # by using keywords, it is possible to add metadata of the text for later usage
    if kwargs:
        for key, value in kwargs.items():
            sentences_df.loc[:, key] = value

    return sentences_df

def write_word_list(list_of_strings, filename):
    with open(filename, 'w') as file:
        for word in list_of_strings:
            file.write("%s\n" % word)

def read_word_list(filename):
    word_list = []
    with open(filename, 'r') as file:
        for word in file:
            word_list.append(word[:-1])
    return word_list