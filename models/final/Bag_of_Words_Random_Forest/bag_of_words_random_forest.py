import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("../../../lib") # Adds higher directory to python modules path.
import helper_functions as hf
import language_processing as lp
import feature_extraction as fe
import price_classifier
from sklearn.model_selection import train_test_split
import nltk
nltk.download('punkt')


PATH = "../../../data/"
data_full = pd.read_csv(f'{PATH}train.tsv', sep='\t')
stopwords = [line.rstrip('\n') for line in open(f'{PATH}stopwords/english')]

# Replace NaN
data_full = hf.replace_nan(data_full)

# Work with only 'Electronics/Video Games & Consoles/Video Gaming Merchandise' category
cat_df = data_full.loc[data_full.category_name == 'Electronics/Video Games & Consoles/Video Gaming Merchandise']

# Delete items without description
cat_df = cat_df[cat_df.item_description != 'No description yet']

# Extract labels
pc = price_classifier.PriceClassifier(cat_df, 5)
y = pc.extract(cat_df)

for range_ in pc.ranges:
    print("range {} has {} items".format(range_, (y == (str(range_[0]) + "-" + str(range_[1]))).sum()))

# Train & Test split
df_train, df_test, y_train, y_test = train_test_split(cat_df, y, test_size=0.20, random_state=42)




# Run grid search
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

NLP_param_grid = {
    'desc_stem': [True, False], 'desc_norm': [True, False], 'desc_sw': [stopwords, None],
    'name_stem': [True, False], 'name_norm': [True, False], 'name_sw': [stopwords, None],
}

grid_search_param_grid = {
    'n_estimators': [1, 3, 10, 30, 100, 300, 1000],
    # 'max_depth':[1, 3, 10, 30, 100],  # first run
    'max_depth': [100, 300, 1000],  # second run
    'min_samples_leaf': [1, 3, 10],
    'random_state': [42]
}

optimisation_result = None

for gp in tqdm(list(ParameterGrid(NLP_param_grid))):

    # Create pipeline
    pipe = fe.Pipeline(steps=[
        ('item_desc', lp.CountVectorizer(
            df_train,
            'item_description',
            stem=gp['desc_stem'],
            normalize=gp['desc_norm'],
            stopwords=gp['desc_sw']
        )),
        ('name', lp.CountVectorizer(
            df_train,
            'name',
            stem=gp['name_stem'],
            normalize=gp['name_norm'],
            stopwords=gp['name_sw']
        ))
    ])
    # Exctract features
    X_train = pipe.extract_features(df_train)
    X_test = pipe.extract_features(df_test)

    # Run grid search
    random_forest = RandomForestClassifier()
    clf = GridSearchCV(random_forest, grid_search_param_grid, iid=False, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)

    current_result = {
        'best_score_': clf.best_score_,
        'best_NLP_gp': gp,
        'best_params_': clf.best_params_,
        'best_estimator_': clf.best_estimator_,
        'X_train': X_train,
        'X_test': X_test,
        # 'pipe': pipe,
    }

    if optimisation_result is None:
        optimisation_result = current_result
        continue

    if current_result['best_score_'] > optimisation_result['best_score_']:
        print("New best score achieved", current_result['best_score_'])
        optimisation_result = current_result

# Add the parameters for grid search and the training data
print("Optimization done, adding additional data into optimisation_result before saving pickle...")
optimisation_result['NLP_param_grid'] = NLP_param_grid
optimisation_result['grid_search_param_grid'] = grid_search_param_grid
optimisation_result['df_train'] = df_train
optimisation_result['df_test'] = df_test
optimisation_result['y_train'] = y_train
optimisation_result['y_test'] = y_test

print("Optimization completed")

print("Saving pickle")
hf.save_pickle(optimisation_result, f'{PATH}pickle/second_BOW_optimisation_result')
print("Done")