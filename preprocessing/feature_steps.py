from sklearn.preprocessing import FunctionTransformer

def add_petal_area(X):
    X = X.copy()
    X["petal_area"] = X["PetalLengthCm"] * X["PetalWidthCm"]
    return X

def add_sepal_area(X):
    X = X.copy()
    X["sepal_area"] = X["SepalLengthCm"] * X["SepalWidthCm"]
    return X
petal_area_transform = FunctionTransformer(add_petal_area)
sepal_area_transform = FunctionTransformer(add_sepal_area)
