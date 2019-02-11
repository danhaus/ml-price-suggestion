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

print("Following functions has been loaded:\n")
print("\
replace_nan\n\
rmse\n\
")
