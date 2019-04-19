from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from sklearn.decomposition import PCA


class Tokenizer():

    def __init__(self, stem, stopwords):
        """
        stemming: Boolean, if words are processed, otherwise they are just converted to lowercase
        """
        self.stem = stem
        self.stopwords = stopwords if stopwords is not None else []
        if self.stem:
            self.ps = PorterStemmer()
            self.stemmed_stopwords = [self.ps.stem(sw) for sw in self.stopwords]

    def tokenize(self, df, column_name):
        """
        df: DataFrame that contains column with text to be tokenized
        column_name: name of the column to be tokenized (string)
        Stems and tokenizes all the text in the item description.
        Returns a dictionary that maps item id to a dictionary storing
        the processed words and their respective count (for the given item)
        """

        # Construct dictionary with processed words for every item in the category
        processed_tokens_d = {}  # dict to store words and their count (as dict) under item id (train_id)
        # Iterate through the items
        stopwords = self.stopwords
        stemmed_stopwords = self.stemmed_stopwords if self.stem else None
        for id_, text in df[column_name].iteritems():
            # Iterate through the sentences
            words_d = {}
            for sentence in sent_tokenize(text):
                # Iterate through the words
                for word in word_tokenize(sentence):
                    if self.stem:
                        processed_word = self.ps.stem(word)
                        if processed_word in stemmed_stopwords:
                            continue
                    else:
                        processed_word = word.lower()
                        if processed_word in stopwords:
                            continue

                    if processed_word not in words_d.keys():
                        words_d[processed_word] = 1
                    else:
                        words_d[processed_word] += 1
            processed_tokens_d[id_] = words_d
        # self.processed_tokens_d = processed_tokens_d # DEBUGGING
        return processed_tokens_d

    # TODO: Implement stopwords

    def create_voc_set(self, processed_tokens_d):
        """
        Create set of all the processed words
        """
        voc_set = set()
        for item in processed_tokens_d.values():
            for word in item:
                voc_set.add(word)
        return voc_set


class CountVectorizer(Tokenizer):
    """
    Class to implement bag of words as features.
    """

    def __init__(self, df_train, column_name, stem, normalize, stopwords):
        """
        df_train: DataFrame to be processed to create vocabulary set whose
            content will be used for tokenizing
        normalize: Boolean, if True, each value of bag of words will be divided
            by number of words for the item it belongs to
        column_name: name of the column containg text to be tokenized
        stopwords: list of stopwords or None
        """
        self.df_train = df_train
        self.normalize = normalize
        self.stem = stem
        self.stopwords = stopwords
        super().__init__(self.stem, self.stopwords)
        self.column_name = column_name
        self.train_processed_tokens = self.tokenize(df_train, self.column_name)
        self.voc_set = self.create_voc_set(self.train_processed_tokens)  # keep set to speed up look ups
        self.voc_set_lst = list(self.voc_set)  # this is the base for the word vectors

    def extract(self, df):
        # Create vocabulary set
        voc_set_lst = self.voc_set_lst
        pre_name = "cv" + "_" + self.column_name + "_"
        columns = [pre_name + word for word in voc_set_lst]
        X = pd.DataFrame(0, index=df.index, columns=columns, dtype='float32')
        processed_tokens_d = self.tokenize(df, self.column_name)
        # Iterate through the items (ids)
        for id_, words in processed_tokens_d.items():
            # Get total number of words for given item as a normalisation factor
            if self.normalize:
                word_len = 0
                for word, count in words.items():
                    word_len += count
            else:
                word_len = 1
            # Iterate through words and their respective counts
            for word, count in words.items():
                # If word in vocab, extract its count, otherwise do nothing
                if word in self.voc_set:
                    X.at[id_, pre_name + word] = count / word_len
        return X


class MeanEmbeddingVectorizer(Tokenizer):
    """
    Class to implement mean embedding as features as follows:
    implementation is simmilar to the bag of words above, but instead of using
    the word frequency, this uses mean of a word vector (e.g. mean over 300
    dimension)
    """

    def __init__(self, model, df_train, column_name):
        """
        model: trained word2vec model (usually stored in .word2vec file) loaded
            using gensim
        df_train: DataFrame to be processed to create vocabulary set whose
            content will be used for tokenizing
        column_name: name of the column containg text to be tokenized
        """
        super().__init__(stem=False)  # initialize the Tokenizer without stemming
        self.model = model
        self.df_train = df_train
        self.column_name = column_name
        self.train_processed_tokens = self.tokenize(self.df_train, self.column_name)
        self.voc_set_intersect = self.create_intersect_voc_set()
        self.voc_set_intersect_lst = list(self.voc_set_intersect)

    def extract(self, df):
        voc_set_lst = self.voc_set_intersect_lst
        pre_name = "mev" + "_" + self.column_name + "_"
        columns = [pre_name + word for word in voc_set_lst]
        X = pd.DataFrame(0, index=df.index, columns=columns, dtype='float32')
        processed_tokens_d = self.tokenize(df, self.column_name)
        # Iterate through the items (ids)
        for id_, words in processed_tokens_d.items():
            # Iterate through words and their respective counts
            for word, count in words.items():
                if word in self.voc_set_intersect:
                    X.at[id_, pre_name + word] = self.model[word].mean()
        return X

    ### Helper methods ###

    def create_intersect_voc_set(self):
        """
        Creates a vocabulary set with only the words that are in both the model and df_train
        """
        voc_set_df_train = self.create_voc_set(self.train_processed_tokens)
        self.voc_set_df_train = voc_set_df_train  # for analysis
        voc_set_model = set(self.model.vocab.keys())  # set from model's vocabulary
        self.voc_set_model = voc_set_model  # for analysis
        voc_intersect = voc_set_df_train.intersection(voc_set_model)  # get intersection of the two sets
        return voc_intersect


class PrincipalAxesExtractor(Tokenizer):
    """
    Tokenizes text, and uses word2vec pretrain model to create a matrix
    with a shape of (n_of_model_dimensions, n_of_words). Then it applies SVD
    to find the most important directions (vectors) of variation. Next, it uses
    these directions as features as follows: 1) select N most important directions
    of variation, 2) unroll them to create columns / features, 3) populate
    the columns with the data for each item / row. (The columns goes as:
    vector1_dir1, ...vector1_dirN, vector2_dir1, ...)
    """

    def __init__(self, model, n_directions, column_name):
        """
        model: trained word2vec model (usually stored in .word2vec file) loaded
            using gensim
        n_directions: number of most significant directions in variation to select
            (most significat vectors from the U matrix after SVD decomposition)
        column_name: name of the column containg text to be processed
        """
        super().__init__(stem=False)  # initialize the Tokenizer without stemming
        self.model = model
        self.n_directions = n_directions
        self.column_name = column_name
        self.pca = PCA(n_components=n_directions)
        self.output_column_name_patern = 'princ_axis_{}_dim_{}'
        self.output_column_names = self.create_column_names(self.output_column_name_patern)

    def extract(self, df):
        pca = self.pca
        output_column_name_patern = self.output_column_name_patern
        n_directions = self.n_directions
        X = pd.DataFrame(0, index=df.index, columns=self.output_column_names, dtype='float32')
        # Get word vectors
        word_vectors_dict = self.create_word_vectors(df)

        for id_, words in word_vectors_dict.items():
            # DataFrame shape: (Nth vector component, word)
            # number of rows = model.vector_size
            # number of columns = number of words
            word_vectors_df = pd.DataFrame.from_dict(words)
            # If the number of words is too small to perform pca, leave all the values
            # set to zeros and skip to the next id_
            if word_vectors_df.shape[1] < n_directions:
                continue
            word_vectors_df_transposed = word_vectors_df.transpose()
            pca.fit(word_vectors_df_transposed)
            X.loc[id_] = pca.components_.ravel()
        return X

    ### Helper methods ###

    def create_word_vectors(self, df):
        model = self.model
        processed_tokens_d = self.tokenize(df, self.column_name)
        word_vectors_dict = {}  # {item_id: {word: word_vector}} only for words in the model voc
        for id_, words in processed_tokens_d.items():
            word_vectors_dict[id_] = {}
            for word in words.keys():
                # Only get the word vector if it is in the model vocabulary
                if word in model.vocab:
                    word_vectors_dict[id_][word] = model[word]
        # self.word_vectors = word_vectors_dict # DEBUGGING
        return word_vectors_dict

    def create_column_names(self, output_column_name_patern):
        n_directions = self.n_directions
        vector_size = self.model.vector_size  # number of dimensions of the model voc
        output_column_names = []
        for comp_n in range(n_directions):
            for dim in range(vector_size):
                output_column_names.append(output_column_name_patern.format(comp_n, dim))
        return output_column_names
