import pandas as pd

# Pipeline infrastructure and classes to make methods for feature extraction
# unified and easily reusable


#### Pipeline infrastructure ###

class Pipeline(object):
    """
    Pipeline that extracts features using all provided feature extractors.
    It is expected to work with pandas DataFrames
    """
    def __init__(self, feature_extractors):
        """
        feature_extractors: list of individual feature extractors which are
            classes containing extract() method
        """
        # list of feature extractor objects
        self.feature_extractors = feature_extractors

    def extract_features(df):
        """
        Extract features using all feature extractors provided to this class.
        df: pandas DataFrame with existing features, extracted features will
            be added to this
        Returns: pandas DataFrame with both passed and extracted features
        """
        df_extracted = None
        for feature_extractor in self.feature_extractors:
            df_extracted = feature_extractor.extract(df_extracted)
        return df_extracted



### Feature extractors ###
# Each feature extractor is expected to work only on one category (some may work on more)
# Each feature extractor must contain extract method that returns

class BaseFeatureExtractor(object):

    def __init__(self):
        pass

    def extract_features(self, df):

        X = pd.DataFrame(index=df.index)

        # Length of name (title)
        X['name_len'] = self.get_len(df, 'name')

        # Length of item description
        X['item_description_len'] = self.get_len(df, 'item_description')

        return X


    ### Help functions ###

    def get_len(self, df, column_name):
        return df[column_name].str.len()
