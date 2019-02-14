def replace_nan(df):
	"""
	Replaces Nan in:
		category_name by 'No Category'
		brand_name by 'No Brand'
		item_description by 'No description yet' (there are already many items with this text)
	Returns new dataframe with these replacements
	"""
	df_c = df.copy()
	import pandas as pd
	df_c.category_name = df_c.category_name.fillna('No Category')
	df_c.brand_name = df_c.brand_name.fillna('No Brand')
	df_c.item_description = df_c.item_description.fillna('No description yet')
	return df_c

def rmse(y_test, y_pred):
	"""
	Returns root mean sqeuared error of two vectors / numbers.
	"""
	import np.sqrt
	from sklearn.metrics import mean_squared_error
	return np.sqrt(mean_squared_error(y_test, y_pred))

def extract_n_random_cats(df, n, random_seed=None):
	"""
	Parameters:
		df: dataframe
		n: number of random categories
		random_seed: random seed to make the category selection predictiable (the same every run)
	"""
	import numpy as np
	unique_cats = df.category_name.unique()
	np.random.seed(random_seed)
	selected_cats = np.random.choice(unique_cats, size=n, replace=False)
	return df.loc[df['category_name'].isin(selected_cats)]


print("Following functions has been loaded:\n")
print("\
replace_nan\n\
rmse\n\
extract_n_random_cats\n\
tokenize\n\
")
