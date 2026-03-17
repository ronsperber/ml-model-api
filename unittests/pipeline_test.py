import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


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

