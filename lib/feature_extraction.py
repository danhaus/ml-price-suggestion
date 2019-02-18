import pandas as pd

# Pipeline infrastructure and classes to make methods for feature extraction
# unified and easily reusable


#### Pipeline infrastructure ###

class Pipeline(object):
    """
    Pipeline that extracts features using all provided feature extractors.
    It is expected to work with pandas DataFrames
    """
    def __init__(self, steps):
        """
        steps: list of (name, feature_extractor) tuples (implementing extract) that are
            chained and whose extract method is executed in that order
        """
        # list of feature extractor objects
        self.steps = steps
        self.named_steps = dict(self.steps)

    def extract_features(self, df):
        """
        Extract features using all feature extractors provided to this class.
        df: pandas DataFrame with existing features, extracted features will
            be added to this
        Returns: pandas DataFrame with both passed and extracted features
        """
        X_lst = []
        for (name, feature_extractor) in self.steps:
            X_lst.append(feature_extractor.extract(df))
        return pd.concat(X_lst, axis=1, sort=False)



### Feature extractors ###
# Each feature extractor is expected to work only on one category (some may work on more)
# Each feature extractor must contain extract method that returns

class BaseFeatureExtractor(object):

    def __init__(self):
        pass

    def extract(self, df):

        X = pd.DataFrame(index=df.index)

        # Length of name (title)
        X['name_len'] = self.get_len(df, 'name')

        # Length of item description
        X['item_description_len'] = self.get_len(df, 'item_description')

        # Item condition & shipping: copy item_condition_id
        X[['item_condition_id', 'shipping']] = df.loc[:,['item_condition_id', 'shipping']]

        return X


    ### Help functions ###

    def get_len(self, df, column_name):
        return df[column_name].str.len()
