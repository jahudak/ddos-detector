import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, average_precision_score, f1_score
)

from catboost import CatBoostClassifier
import wandb

df_a_events = pd.read_parquet("data/A_events.parquet")
df_a_components = pd.read_parquet("data/A_components.parquet")
df_b_events = pd.read_parquet("data/B_events.parquet")
df_b_components = pd.read_parquet("data/B_components.parquet")
df_c_events = pd.read_parquet("data/C_events.parquet")
df_c_components = pd.read_parquet("data/C_components.parquet")
df_d_events = pd.read_parquet("data/D_events.parquet")
df_d_components = pd.read_parquet("data/D_components.parquet")

def merge_events_components(events_df, components_df):
    # Drop common columns from events_df (except 'Attack ID')
    common_columns = set(events_df.columns) & set(components_df.columns)
    common_columns.discard("Attack ID")

    events_df_clean = events_df.drop(columns=common_columns)

    # Merge on 'Attack ID'
    merged_df = pd.merge(
        components_df,
        events_df_clean,
        on="Attack ID",
        how="inner"
    )

    return merged_df

df_a = merge_events_components(df_a_events, df_a_components)
df_b = merge_events_components(df_b_events, df_b_components)
df_c = merge_events_components(df_c_events, df_c_components)
df_d = merge_events_components(df_d_events, df_d_components)

df = pd.concat([df_a, df_b], ignore_index=True)

print("Data merge complete")

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unnecessary columns
    df = df.drop(columns=["Card", "Significant flag", "Whitelist flag", "Attack code", "Detect count"], errors='ignore')

    # Filter out rows where Start time or End time is "0"
    df = df[
        (df["Start time"] != "0") &
        (df["End time"] != "0")
    ]

    # Convert to datetime and extract time features
    df["Start time"] = pd.to_datetime(df["Start time"])
    df["End time"] = pd.to_datetime(df["End time"])

    df["Start_time_hour"] = df["Start time"].dt.hour
    df["Start_time_weekday"] = df["Start time"].dt.weekday
    df["Start_time_dayofyear"] = df["Start time"].dt.dayofyear

    df["End_time_hour"] = df["End time"].dt.hour
    df["End_time_weekday"] = df["End time"].dt.weekday
    df["End_time_dayofyear"] = df["End time"].dt.dayofyear

    # Helper for cyclical encoding
    def sin_cos_encode(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
        radians = 2 * np.pi * series / period
        return np.sin(radians), np.cos(radians)

    # Apply cyclical encoding
    df["Start_time_hour_sin"], df["Start_time_hour_cos"] = sin_cos_encode(df["Start_time_hour"], 24)
    df["Start_time_weekday_sin"], df["Start_time_weekday_cos"] = sin_cos_encode(df["Start_time_weekday"], 7)
    df["Start_time_dayofyear_sin"], df["Start_time_dayofyear_cos"] = sin_cos_encode(df["Start_time_dayofyear"], 365)

    df["End_time_hour_sin"], df["End_time_hour_cos"] = sin_cos_encode(df["End_time_hour"], 24)
    df["End_time_weekday_sin"], df["End_time_weekday_cos"] = sin_cos_encode(df["End_time_weekday"], 7)
    df["End_time_dayofyear_sin"], df["End_time_dayofyear_cos"] = sin_cos_encode(df["End_time_dayofyear"], 365)

    # Extract numeric part from Victim IP
    df["Victim IP Number"] = df["Victim IP"].str.extract(r"IP_(\d+)").astype(int)

    # Preserve original Type column
    df["Type_Original"] = df["Type"]

    # One-hot encode the Type column
    df = pd.get_dummies(df, columns=['Type'])

    # Rename the original Type column back
    df = df.rename(columns={"Type_Original": "Type"})

    # Rename columns for consistency
    df = df.rename(columns={
        "Victim IP Number": "victim_ip",
        "Port number": "port_number",
        "Packet speed": "packet_speed",
        "Data speed": "data_speed",
        "Avg packet len": "packet_len",
        "Source IP count": "source_ip",
        "Start_time_hour_sin": "start_hour",
        "Start_time_weekday_sin": "start_weekday",
        "Start_time_dayofyear_sin": "start_dayofyear",
        "End_time_hour_sin": "end_hour",
        "End_time_weekday_sin": "end_weekday",
        "End_time_dayofyear_sin": "end_dayofyear",
        "Type_DDoS attack": "type_ddos",
        "Type_Normal traffic": "type_normal",
        "Type_Suspicious traffic": "type_sus"
    })

    # Select final set of columns
    selected_columns = [
        "victim_ip", "port_number", "packet_speed", "data_speed", "packet_len", "source_ip",
        "start_hour", "start_weekday", "start_dayofyear",
        "end_hour", "end_weekday", "end_dayofyear",
        "type_ddos", "type_normal", "type_sus"
    ]
    df = df[selected_columns]

    # Refactor: consolidate type columns into single label
    df["type"] = df[["type_ddos", "type_sus", "type_normal"]].idxmax(axis=1)
    df["type"] = df["type"].map({
        "type_ddos": "ddos",
        "type_sus": "sus",
        "type_normal": "normal"
    })

    # Drop the one-hot columns after mapping
    df = df.drop(columns=["type_ddos", "type_sus", "type_normal"])

    return df

df = preprocess_df(df)
df_eval = preprocess_df(df_c)
df_gen = preprocess_df(df_d)

print("Data preprocess complete")

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "f1_macro",
        "goal": "maximize"
    },
    "parameters": {
        "iterations": {
            "values": [100, 300, 500]
        },
        "learning_rate": {
            "min": 0.01,
            "max": 0.2
        },
        "depth": {
            "values": [4, 6, 8, 10]
        },
        "l2_leaf_reg": {
            "min": 1,
            "max": 10
        },
        "bagging_temperature": {
            "min": 0.0,
            "max": 1.0
        },
        "random_strength": {
            "min": 0,
            "max": 10
        }
    }
}

def sweep(df: pd.DataFrame = df, df_eval: pd.DataFrame = df_eval) -> dict:
    wandb.init(project="hamlab")

    config = wandb.config

    feature_cols = [col for col in df.columns if col != "type"]
    X_train = df[feature_cols].copy()
    y_train = df["type"]

    X_eval = df_eval[feature_cols].copy()
    y_eval = df_eval["type"]

    classes = ["ddos", "normal", "sus"]
    y_eval_bin = label_binarize(y_eval, classes=classes)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    class_counts = y_train.value_counts(normalize=True).to_dict()
    inverse_class_weights = {k: 1 / v for k, v in class_counts.items()}
    total_weight = sum(inverse_class_weights.values())
    normalized_weights = {k: (inverse_class_weights[k] / total_weight) * len(classes) for k in inverse_class_weights}
    class_weights_list = [normalized_weights.get(c, 1.0) for c in classes]

    clf = CatBoostClassifier(
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        bagging_temperature=config.bagging_temperature,
        random_strength=config.random_strength,
        loss_function="MultiClass",
        verbose=50,
        random_state=42,
        od_type="Iter",
        od_wait=20,
        task_type="CPU",
        thread_count=-1,
        class_weights=class_weights_list
    )

    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_eval_scaled).ravel()
    y_prob = clf.predict_proba(X_eval_scaled)

    f1 = f1_score(y_eval, y_pred, average="macro")
    roc_auc = roc_auc_score(y_eval_bin, y_prob, average="macro", multi_class="ovr")
    pr_auc = average_precision_score(y_eval_bin, y_prob, average="macro")

    wandb.log({
        "f1_macro": f1,
        "roc_auc_macro": roc_auc,
        "pr_auc_macro": pr_auc
    })

    wandb.finish()

    return {
        "f1_macro": f1,
        "roc_auc_macro": roc_auc,
        "pr_auc_macro": pr_auc
    }

print("Starting sweep...")

sweep_id = wandb.sweep(sweep_config, project="hamlab")
wandb.agent(sweep_id, function=sweep, count=20)