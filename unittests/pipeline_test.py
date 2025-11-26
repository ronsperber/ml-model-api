import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from preprocessing_steps.iris_test import iris_test_preprocessing_steps
from preprocessing_steps.iris_test import clean_df, add_petal_area, add_sepal_area

def test_test_clean():
    df = pd.DataFrame({
        "A": [1, 2],
        "B": [3, 4],
        "Species": ["X", "Y"]
    })

    out = clean_df(df)
    # numeric columns should be multiplied by 2 and +1
    assert out["A"].tolist() == [3, 5]
    assert out["B"].tolist() == [7, 9]
    # ensure Species untouched
    assert out["Species"].tolist() == ["X", "Y"]

def test_add_petal_area():
    df = pd.DataFrame({
        "PetalLengthCm": [1, 2],
        "PetalWidthCm": [3, 4]
    })
    out = add_petal_area(df)
    assert "petal_area" in out.columns
    assert out["petal_area"].tolist() == [3, 8]

def test_add_sepal_area():
    df = pd.DataFrame({
        "SepalLengthCm": [1, 2],
        "SepalWidthCm": [3, 4]
    })
    out = add_sepal_area(df)
    assert "sepal_area" in out.columns
    assert out["sepal_area"].tolist() == [3, 8]

def test_feature_steps_pipeline():
    df = pd.DataFrame({
        "PetalLengthCm": [1],
        "PetalWidthCm": [2],
        "SepalLengthCm": [3],
        "SepalWidthCm": [4],
    })

    # build a mini pipeline of just your feature steps
    pipe = Pipeline(iris_test_preprocessing_steps)
    out = pipe.fit_transform(df)

    assert "petal_area" in out.columns
    assert "sepal_area" in out.columns
    cleaned_sepal = df["SepalLengthCm"]*2 + 1, df["SepalWidthCm"]*2 + 1
    cleaned_petal = df["PetalLengthCm"]*2 + 1, df["PetalWidthCm"]*2 + 1
    expected_sepal_area = cleaned_sepal[0] * cleaned_sepal[1]
    expected_petal_area = cleaned_petal[0] * cleaned_petal[1]

    assert out["petal_area"].iloc[0] == expected_petal_area.iloc[0]
    assert out["sepal_area"].iloc[0] == expected_sepal_area.iloc[0]



def test_column_transformer():
    df = pd.DataFrame({
        "num1": [1,2],
        "cat1": ["A","B"]
    })

    ct = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["cat1"]),
            ("num", "passthrough", ["num1"])
        ]
    )

    out = ct.fit_transform(df)

    # Expect shape == 2 rows × (2 categories + 1 numeric) columns
    assert out.shape == (2, 3)

def test_pipeline_preprocessing_only():
    df = pd.DataFrame({
    "PetalLengthCm": [1],
    "PetalWidthCm": [2],
    "SepalLengthCm": [3],
    "SepalWidthCm": [4],
    })
    categorical_cols =df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    one_hot = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
            ]
            )
    pipe = Pipeline([
        *iris_test_preprocessing_steps,
        ("process", one_hot)
    ])


    out = pipe.fit_transform(df)

    assert out.shape[0] == len(df)
    assert out.shape[1] > 0
