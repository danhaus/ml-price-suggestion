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
# Each feature extractor must contain extract method that returns
