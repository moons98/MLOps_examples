from airflow.providers.postgres.hooks.postgres import PostgresHook

from pickle import dumps

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import optuna
from optuna.storages import RDBStorage


def train_fn(**context):
    hook = PostgresHook(postgres_conn_id="postgres_default")
    conn = hook.get_conn()
    stmt = """
            SELECT *
              FROM hospital_train
            """
    data = pd.read_sql(stmt, conn)
    label = data["OC"]

    data.drop(columns=["OC", "inst_id", "openDate"], inplace=True)

    x_train, x_valid, y_train, y_valid = train_test_split(
        data, label, test_size=0.3, shuffle=True, stratify=label
    )
    x_train = x_train.reset_index(drop=True)
    x_valid = x_valid.reset_index(drop=True)

    cat_columns = data.select_dtypes(include="object").columns
    num_columns = data.select_dtypes(exclude="object").columns

    print("Categorical columns: ", cat_columns)
    print("Numerical columns: ", num_columns)

    preprocessor = ColumnTransformer(
        transformers=[
            ("impute", SimpleImputer(), num_columns),
            ("scaler", StandardScaler(), num_columns),
            (
                "encoding",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_columns,
            ),
        ]
    )

    le = LabelEncoder()

    y_train = le.fit_transform(y_train)
    y_valid = le.transform(y_valid)

    x_train = preprocessor.fit_transform(x_train)
    x_valid = preprocessor.transform(x_valid)

    # Model Tunes

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 2, 100)
        max_depth = int(trial.suggest_int("max_depth", 1, 32))
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(x_train, y_train)
        return f1_score(y_valid, model.predict(x_valid))

    storage = RDBStorage(url=hook.get_uri().replace("/postgres", "/optuna"))

    study = optuna.create_study(
        study_name="hospital_model",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    best_metric = study.best_value

    print("Best params: ", best_params)
    print("Best metric: ", best_metric)

    # Model Train
    model = RandomForestClassifier(**best_params)
    model.fit(x_train, y_train)

    print("Validation Score: ", f1_score(y_valid, model.predict(x_valid)))

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
