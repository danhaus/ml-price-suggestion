from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd

class StemmedTokenizer():

    def __init__(self):
        pass

    def tokenize(self, df, column_name):
    	"""
        df: DataFrame that contains column with text to be tokenized
        column_name: name of the column to be tokenized (string)
    	Stems and tokenizes all the text in the item description.
    	Returns a dictionary that maps item id to a dictionary storing the stemmed words and their count (for the given item)
    	"""
    	from nltk.stem import PorterStemmer
    	from nltk.tokenize import sent_tokenize, word_tokenize
    	ps = PorterStemmer()

    	# Construct dictionary with stemmed words for every item in the category

    	stemmed_tokens_d = {} # dict to store words and their count (as dict) under item id (train_id)
    	# Iterate through the items
    	for id_, text in df[column_name].iteritems():
    		# Iterate through the sentences
    		for sentence in sent_tokenize(text):
    			# Iterate through the words
    			words_d = {}
    			for word in word_tokenize(sentence):
    				stemmed_word = ps.stem(word)
    				if stemmed_word not in words_d.keys():
    					words_d[stemmed_word] = 1
    				else:
    					words_d[stemmed_word] += 1
    			stemmed_tokens_d[id_] = words_d
    	return stemmed_tokens_d



class CountVectorizer(StemmedTokenizer):

    def __init__(self, df_train, column_name, normalize):
        """
        df_train: DataFrame to be processed to create vocabulary set whose
            content will be used for tokenizing
        normalize: Boolean, if True, each value of bag of words will be divided
            by number of words for the item it belongs to
        column_name: name of the column containg text to be tokenized
        """
        self.df_train = df_train
        self.normalize = normalize
        self.column_name = column_name
        self.train_stemmed_tokens = self.tokenize(df_train, self.column_name)
        self.voc_set = self.create_voc_set() # keep set to speed up look ups
        self.voc_set_lst = list(self.voc_set) # this is the base for the word vectors

    def extract(self, df):
        # Create vocabulary set
        voc_set_lst = self.voc_set_lst
        columns = [self.column_name + "_" + word for word in voc_set_lst]
        X = pd.DataFrame(0, index=df.index, columns=columns, dtype='float32')
        stemmed_tokens_d = self.tokenize(df, self.column_name)
        # Iterate through the items (ids)
        for id_, words in stemmed_tokens_d.items():
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
                    X.at[id_, self.column_name + "_" + word] = count / word_len
        return X

    def create_voc_set(self):
        """
        Create set of all the stemmed words
        """
        stemmed_tokens_d = self.train_stemmed_tokens
        voc_set = set()
        for item in stemmed_tokens_d.values():
            for word in item:
                voc_set.add(word)
        return voc_set
