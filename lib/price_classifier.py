import pandas as pd
import numpy as np

class PriceClassifier(object):
    """
    Takes pandas DataFrame and a number of classes and creates classes for
    prices with an inted to have equal number of samples in each class
    (not always possible). Setting the bounds for the classes is hence
    done dynamically with an intend to equalise the area under histogram
    for each class.
    """

    def __init__(self, df, n_classes):
        self.ranges = self.make_price_ranges(df, n_classes)


    def make_price_ranges(self, df, n_classes):
        prices = df.price
        sorted_prices = prices.sort_values()
        l = len(df)
        n = int(l / (n_classes)) # ideal number of items in each class (range)

        ranges = [] # list of tuples
        counter = 0
        lower = sorted_prices.iloc[0]
        upper = None
        for index, val in sorted_prices.iteritems():
            counter += 1
            if counter % n == 0:
                upper = val
                ranges.append((lower, upper))
                lower = upper

        # Extend the lower bound of the first range to 0
        ranges[0] = (0, ranges[0][1])
        # Extend the upper bound of the last range to infinity
        ranges[-1] = (ranges[-1][0], np.inf)
        return ranges

    def extract(self, df):
        """
        Extract classes from the DataFrame and return it as Series.
        """
        prices = df.price
        price_classes = pd.Series(index=df.index)

        for range_ in self.ranges:
            lower_bound = range_[0]
            upper_bound = range_[1]
            ids = prices.loc[(prices >= lower_bound) & (prices < upper_bound)].index
            price_classes.loc[ids] = '{}-{}'.format(lower_bound, upper_bound)


        return price_classes
