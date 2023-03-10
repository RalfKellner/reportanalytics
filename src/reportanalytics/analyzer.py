import os
import pickle
import nltk
import umap
import hdbscan
import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import dbscan
from sklearn.preprocessing import normalize
from .utils import read_word_list
import dropbox
dbx = dropbox.Dropbox('sl.BaRU0C4tvubKSzMOm7SjNFDhVygoNbDEEybEqh9-5nvJ7AC4-_BE0meGdfMvk7wW703NHI0kE91ckQBh1nTZF5MjNIPRsWyx26_yAQiAc8lwfzphkMnRDlw4lwZqREAcFJR-D9s')

class ReportAnalysis:

    def __init__(self, doc_model_name = '10K-long'):
        self.doc_model = self._load_doc_model(doc_model_name)
        self.word_vectors = self.doc_model.wv.get_normed_vectors()
        self.vocab = list(self.doc_model.wv.key_to_index.keys())
        self.word_list = self._load_word_list()


    def fit_topic_model(self, document_vectors, ntopic_words = 25, umap_args = None, hdbscan_args = None, rm_docs = False, cosine_threshold = 0.50, n_reduced_topics = None):
        
        '''
            This method uses a set of l2-normalized embedded document vectors, reduces their dimension by the UMAP dimensionality reduction model, 
            then the HDBSCAN cluster model to generate clusters which are considered to represent the topics, however, topic vectors are calculated using
            the centroid of embedded document vectors per cluster in the original embedding space. In addition, the number of topics is reduced
            by a cosine similarity threshold. After the process is finished, two sets of topic vectors and topic words are available. The one with the original
            topic number from the HDBSCAN algorithm and the one with a reduced number of topics.

            Arguments:

            document_vectors [numpy.array]: a numpy array with l2-normalized document vector embeddings which match to the class doc2vec model
            ntopic_words [int]: the number of words which are most similar to the topic
            umap_args [dict]: a dictionary with parameters for the UMAP model, default=None using the internal default arguments
            hdbscan_args [dict]: a dictionary with parameters for the HDBSCAN model, default=None using the internal default arguments
            rm_docs [logical]: Save the document vectors which have been used to train the topics if True, default = False which means vectors are deleted 
            after topics have been trained
            cosine_threshold [float]: a value between 0 and 1 which defines that original topic vectors which are similar higher than this threshold
            n_reduced_topics [int]: can be used as an alternative to the cosine similarity logic for topic reduction, if a fixed number of topics is desired;
            this should be done with caution, because topics with low similarity might be merged
        '''
        
        assert (cosine_threshold == None) + (n_reduced_topics == None) == 1, 'cosine_threshold or n_reduced_topics should be speficied, not both at the same time. The second argument must be None.' 
        if cosine_threshold:
            assert 0.10 <= cosine_threshold <= 0.90, 'cosine_threshold should lie in the range [0.10, 0.90] to get reasonable reduced topics.'
        else:
            assert isinstance(n_reduced_topics, int) and n_reduced_topics > 0, 'the number of n_reduced_topics must be a positive integer value'

        self._validate_normed_vectors(document_vectors, 'document')        
        self.document_vectors = document_vectors
        self.ntopic_words = ntopic_words

        if umap_args == None:
            umap_args = {'n_neighbors': 15,
                'n_components': 5,
                'metric': 'cosine'}

        if hdbscan_args == None:
            hdbscan_args = {'min_cluster_size': 50,
                    'metric': 'euclidean',
                    'cluster_selection_method': 'eom'}
            
        print('Starting to train the dimensionality reduction model...')
        self.umap_model = umap.UMAP(**umap_args).fit(self.document_vectors)
        print('Umap model for dimensionality reduction has been trained!\n')

        print('Starting to train the clustering model...')
        self.cluster_model = hdbscan.HDBSCAN(**hdbscan_args).fit(self.umap_model.embedding_)
        print('HDBSCAN model for clustering has been trained!\n')

        print('Creating topic vectors...')
        self._create_topic_vectors(self.cluster_model.labels_)
        self._deduplicate_topics()
        print(f'Creating topic vectors finished! {self.topic_vectors.shape[0]} topics have been found.')

        self.topic_words, self.topic_word_scores = self._find_topic_words_and_scores(topic_vectors=self.topic_vectors, nwords = self.ntopic_words)
        print('Creating a reduced topic model...')
        self._reduce_topic_vectors(cosine_threshold = cosine_threshold, n_reduced_topics = n_reduced_topics)
        print(f'Topic reduction is finished, the number of reduced topics is {self.reduced_topic_vectors.shape[0]}')
        if rm_docs:
            self.document_vectors = None


    def find_close_docs_to_words(self, document_vectors, words = 'esgwords', topn_documents = 5):

        self._validate_words(words)
        not_in_vocab, in_vocab = self._wordlist_prepared(words)
    
        normed_word_vectors = np.array([self._l2_normalize(self.doc_model.wv[word]) for word in in_vocab])
        res = np.inner(normed_word_vectors, document_vectors)
        top_docs = np.flip(np.argsort(res, axis = 1), axis = 1)[:, :topn_documents]
        top_doc_scores = np.flip(np.sort(res, axis = 1), axis = 1)[:, :topn_documents]

        top_docs_df = pd.DataFrame(top_docs, index = in_vocab)
        top_doc_scores_df = pd.DataFrame(top_doc_scores, index = in_vocab)

        return top_docs_df, top_doc_scores_df
    

    def find_close_words_to_docs(self, words, document_vectors, topn_words):

        self._validate_words(words)
        not_in_vocab, in_vocab = self._wordlist_prepared(words)
    
        normed_word_vectors = np.array([self._l2_normalize(self.doc_model.wv[word]) for word in in_vocab])
        res = np.inner(document_vectors, normed_word_vectors)
        top_words = np.flip(np.argsort(res, axis = 1), axis = 1)[:, :topn_words]
        top_word_scores = np.flip(np.sort(res, axis = 1), axis = 1)[:, :topn_words]

        top_words_df = pd.DataFrame(top_words)
        for i in range(top_words_df.shape[0]):
            top_words_df.iloc[i, :] = [in_vocab[idx] for idx in list(top_words_df.iloc[i].values)]
        top_word_scores_df = pd.DataFrame(top_word_scores)

        return top_words_df, top_word_scores_df
    

    def find_close_topics_to_words(self, topic_vectors, words = 'esgwords', topn_topics = 3):

        self._validate_words(words)
        not_in_vocab, in_vocab = self._wordlist_prepared(words)
        
        normed_word_vectors = np.array([self._l2_normalize(self.doc_model.wv[word]) for word in in_vocab])
        res = np.inner(normed_word_vectors, topic_vectors)
        top_topics = np.flip(np.argsort(res, axis = 1), axis = 1)[:, :topn_topics]
        top_topic_scores = np.flip(np.sort(res, axis = 1), axis = 1)[:, :topn_topics]

        top_topics_df = pd.DataFrame(top_topics, index = in_vocab)
        top_topic_scores_df = pd.DataFrame(top_topic_scores, index = in_vocab)

        return top_topics_df, top_topic_scores_df


    def find_close_topics_to_docs(self, document_vectors, topic_vectors, topn_topics = 3):
        
        self._validate_normed_vectors(document_vectors, 'document')
        self._validate_normed_vectors(topic_vectors, 'topic')
        
        res = np.inner(document_vectors, topic_vectors)
        top_topics = np.flip(np.argsort(res, axis = 1), axis = 1)[:, :topn_topics]
        top_topic_scores = np.flip(np.sort(res, axis = 1), axis = 1)[:, :topn_topics]

        top_topics_df = pd.DataFrame(top_topics)
        top_topic_scores_df = pd.DataFrame(top_topic_scores)

        return top_topics_df, top_topic_scores_df
    

    def most_similar_words(self, words, n_words = 5):
        
        self._validate_words(words)
        not_in_vocab, in_vocab = self._wordlist_prepared(words)

        normed_word_vectors = np.array([self._l2_normalize(self.doc_model.wv[word]) for word in in_vocab])
        res = np.inner(normed_word_vectors, self.word_vectors)

        most_sim_idx = np.flip(np.argsort(res, axis = 1), axis = 1)[:, 1:(n_words+1)]
        most_sim_words = [[self.doc_model.wv.index_to_key[idx] for idx in list(idx_array)] for idx_array in list(most_sim_idx)]
        most_sim_scores = np.flip(np.sort(res, axis = 1), axis = 1)[:, 1:(n_words+1)]
        
        most_sim_words_df = pd.DataFrame(most_sim_words, index = in_vocab)
        most_sim_scores_df = pd.DataFrame(most_sim_scores, index = in_vocab)

        return most_sim_words_df, most_sim_scores_df


    def save(self, file):
        #create a pickle file
        with open(file, 'wb') as handle:
            #pickle the dictionary and write it to file
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load(self, file):
        with open(file, 'rb') as handle:
            out = pickle.load(handle)
        return out


    def _find_topic_words_and_scores(self, topic_vectors, nwords):

        topic_words = []
        topic_word_scores = []

        res = np.inner(topic_vectors, self.word_vectors)

        top_words = np.flip(np.argsort(res, axis = 1), axis = 1)
        top_scores = np.flip(np.sort(res, axis = 1), axis = 1)

        for words, scores in zip(top_words, top_scores):
            topic_words.append([self.vocab[i] for i in words[0:nwords]])
            topic_word_scores.append(scores[0:nwords])

        topic_words = np.array(topic_words)
        topic_word_scores = np.array(topic_word_scores)

        return topic_words, topic_word_scores
    

    def _create_topic_vectors(self, cluster_labels):
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        self.topic_vectors = self._l2_normalize(
            np.vstack([self.document_vectors[np.where(cluster_labels == label)[0]]
                      .mean(axis=0) for label in unique_labels]))
        
    
    def _deduplicate_topics(self):
        core_samples, labels = dbscan(X=self.topic_vectors,
                                      eps=0.1,
                                      min_samples=2,
                                      metric="cosine")

        duplicate_clusters = set(labels)

        if len(duplicate_clusters) > 1 or -1 not in duplicate_clusters:

            unique_topics = self.topic_vectors[np.where(labels == -1)[0]]

            if -1 in duplicate_clusters:
                duplicate_clusters.remove(-1)

            for unique_label in duplicate_clusters:
                unique_topics = np.vstack(
                    [unique_topics, self._l2_normalize(self.topic_vectors[np.where(labels == unique_label)[0]]
                                                       .mean(axis=0))])

            self.topic_vectors = unique_topics


    def _reduce_topic_vectors(self, cosine_threshold, n_reduced_topics):

        topic_vectors = self.topic_vectors
        n_topics = topic_vectors.shape[0]
        inner = np.inner(topic_vectors, topic_vectors)
        np.fill_diagonal(inner, 0.0)
        topsim = np.max(inner)

        if cosine_threshold:
            while topsim > cosine_threshold:
                inner = np.inner(topic_vectors, topic_vectors)
                np.fill_diagonal(inner, 0.0)
                topsim = np.max(inner)
                topic_pair, _ = np.where(inner == topsim)
                new_topic = topic_vectors[topic_pair].mean(axis = 0)
                topic_vectors = np.delete(topic_vectors, topic_pair, axis = 0)
                topic_vectors = np.append(topic_vectors, new_topic.reshape(1, -1), axis = 0)
        elif n_reduced_topics:
            while n_topics > n_reduced_topics:
                inner = np.inner(topic_vectors, topic_vectors)
                np.fill_diagonal(inner, 0.0)
                topsim = np.max(inner)
                topic_pair, _ = np.where(inner == topsim)
                new_topic = topic_vectors[topic_pair].mean(axis = 0)
                topic_vectors = np.delete(topic_vectors, topic_pair, axis = 0)
                topic_vectors = np.append(topic_vectors, new_topic.reshape(1, -1), axis = 0)
                n_topics = topic_vectors.shape[0]

        self.reduced_topic_vectors = topic_vectors
        self.reduced_topic_words, self.reduced_topic_word_scores = self._find_topic_words_and_scores(topic_vectors=self.reduced_topic_vectors, nwords = self.ntopic_words)

        print(f'The number of topics has been reduced from {len(self.topic_vectors)} to {len(self.reduced_topic_vectors)}.')


    def _wordlist_prepared(self, words, print_info = True):
        if isinstance(words, str):
            word_list = self.word_list[words]
        elif isinstance(words, list):
            word_list = words

        in_vocab = []
        not_in_vocab = []
        for word in word_list:
            if word in self.vocab:
                in_vocab.append(word)
            else:
                not_in_vocab.append(word)

        if (len(not_in_vocab) > 0) and print_info:
            print('The following words are not in the vocabulary:\n')
            print(not_in_vocab)

        if len(in_vocab) == 0:
            raise ValueError('No word of the word list exists in the Word2Vec model.')
        
        return not_in_vocab, in_vocab
    

    def _load_word_list(self):
        word_list = dict()
        for filename in ["ewords.txt", "swords.txt", "gwords.txt"]:
            this_dir, _ = os.path.split(__file__)
            data_path = os.path.join(this_dir, "files", filename)
            word_list[filename.split(".")[0]] = read_word_list(data_path)
        all_words = []
        for key in word_list.keys():
            all_words += word_list[key]
        word_list['esgwords'] = all_words

        lost_words = []
        for key in word_list.keys():
            not_in_vocab, in_vocab = self._wordlist_prepared(word_list[key], print_info=False)
            word_list[key] = in_vocab
            lost_words.extend(not_in_vocab)
        print(f'In sum {len(set(lost_words))} words from the default word lists could not be found in the Word2Vec vocabulary. These words are deleted from the word list:')
        print(set(lost_words))
        return word_list

    @staticmethod
    def _l2_normalize(vectors):
        if vectors.ndim == 2:
            return normalize(vectors)
        else:
            return normalize(vectors.reshape(1, -1))[0]


    @staticmethod
    def _load_doc_model(model_name):
        this_dir, _ = os.path.split(__file__)
        if model_name == '10K-long':
            data_path = os.path.join(this_dir, "files", "d2v_10K_long.pkl")
            file_name = '/d2v_10K_long.pkl'
        elif model_name == '10K-10Q-short':
            data_path = os.path.join(this_dir, "files", "d2v_10K10Q_short.pkl")
            file_name = '/d2v_10K10Q_short.pkl'
        else:
            raise NameError('Model name must be 10K-long or 10K-10Q-short.')
        
        try:
            with open(data_path, 'rb') as handle:
                model = pickle.load(handle)
        except:
            print('Model needs to be downloaded before first time usage.')
            dbx.files_download_to_file(data_path, file_name)
            with open(data_path, 'rb') as handle:
                model = pickle.load(handle)
        return model
    

    @staticmethod
    def _validate_normed_vectors(vectors, name):
        norm_check = any([not(0.999 <= nbr <= 1.001) for nbr in np.linalg.norm(vectors, axis = 1)])
        if norm_check:
            warnings.warn(f'Not all {name} vectors have a length of 1, make sure they are l2-normalized.')


    @staticmethod
    def _validate_words(words):
        if isinstance(words, str):
            assert words in ['ewords', 'swords', 'gwords', 'esgwords'], 'When using internal word list, words must be "ewords", "swords", "gwords" or "esgwords"'
        elif isinstance(words, list):
            assert all([isinstance(word, str) for word in words]), 'If a user defined word list is provided, a list of strings must be used.'
        else:
            raise TypeError('Either use a string for build-in word lists or provide a list of strings.')
