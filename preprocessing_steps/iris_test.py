from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
def clean_df(df):
    df = df.copy()
    numeric_cols = df.select_dtypes("number").columns
    df[numeric_cols] = df[numeric_cols] * 2 + 1
    return df


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
clean_transform = FunctionTransformer(clean_df)

iris_test_preprocessing_steps = [
    ("clean", clean_transform),
    ("petal_area", petal_area_transform),
    ("sepal_area", sepal_area_transform),
    ("scaling", StandardScaler())
]