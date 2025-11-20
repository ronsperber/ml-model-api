from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

def train(configs: dict) -> dict:
    # load the data
    df = pd.read_csv(configs["dataset_path"])
    # if an index column specified set the index column to be the index
    if configs["index_col"] is not None:
        df = df.set_index(configs["index_col"])
    # apply the cleaning function
    clean_fn = configs.get("clean_fn", lambda x: x)
    df = clean_fn(df)
    # get the feature set
    X = df.drop(columns=[configs["target_col"]])
    y_raw = df[configs["target_col"]]
    # encode the labels numerically
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    # find categorical and numeric columns
    feature_steps = configs.get("feature_steps", [])
    # Detect column types after feature engineering 
    X_tmp = X.copy()
    for _ , transformer in feature_steps:
        X_tmp = transformer.fit_transform(X_tmp)
    categorical_cols = X_tmp.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = X_tmp.select_dtypes(include=["number", "bool"]).columns.tolist()
    # create preprocessor to to one hot encode any non-numeric columns
    preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
    )
    ModelClass = configs["model_type"]
    kwargs = configs.get("model_params", {})
    model_cls = ModelClass(**kwargs)

    pipeline = Pipeline(
    [
        *feature_steps,
        ("process", preprocessor),
        ("model", model_cls),
        *configs.get("pipeline_steps",[])
    ]
    )
    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=configs.get("test_size", 0.2),
        random_state=configs.get("random_state", 42)
    )
    # create the model
    model = GridSearchCV(
        pipeline,
        configs["param_grid"]
    )
    # train model
    print("Training model...")
    model.fit(X_train, y_train)
    feature_names = model.best_estimator_.named_steps["process"].get_feature_names_out()
    print("Training complete.")
    acc = model.score(X_test, y_test)
    # get feature importances if the model records them
    feature_importances = []
    model_estimator = model.best_estimator_["model"]
    if hasattr(model_estimator,"feature_importances_"):
        feature_importances = model_estimator.feature_importances_.tolist()
    metadata = {
        "model_type": configs["model_type"].__name__,
        "features" : feature_names.tolist(),
        "feature_importances" : feature_importances,
        "best_params": model.best_params_,
        "test_acc": acc,
        "classes": list(le.classes_)
    }
    return {
        "model": model,
        "metadata": metadata
    }





    
