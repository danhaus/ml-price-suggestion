
def rmse(y_test, y_pred):
    """
    Returns root mean sqeuared error of two vectors / numbers.
    """
    import np.sqrt
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("Following functions has been loaded:\n")
print("\
rmse\n\
")
