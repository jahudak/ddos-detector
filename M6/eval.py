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

import os
import matplotlib.pyplot as plt
import seaborn as sns

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

df_train = pd.concat([df_a, df_b], ignore_index=True)

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

df_train = preprocess_df(df_train)
df_test = preprocess_df(df_c)
df_val = preprocess_df(df_d)

print("Data preprocess complete")

def eval(df_train: pd.DataFrame, df_test: pd.DataFrame, df_val: pd.DataFrame) -> dict:
    feature_cols = [col for col in df_train.columns if col != "type"]
    X_train = df_train[feature_cols].copy()
    y_train = df_train["type"]

    classes = ["ddos", "normal", "sus"]

    def prepare_eval_data(df):
        X = df[feature_cols].copy()
        y = df["type"]
        y_bin = label_binarize(y, classes=classes)
        return X, y, y_bin

    X_eval, y_eval, y_eval_bin = prepare_eval_data(df_test)
    X_gen, y_gen, y_gen_bin = prepare_eval_data(df_val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)
    X_gen_scaled = scaler.transform(X_gen)

    clf = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1845738069446013,
        depth=10,
        l2_leaf_reg=7,
        bagging_temperature=0.09256507943598458,
        random_strength=0,
        loss_function="MultiClassOneVsAll",
        verbose=50,
        random_state=42,
        od_type="Iter",
        od_wait=20,
        task_type="CPU",
        thread_count=-1
    )

    clf.fit(X_train_scaled, y_train)

    def evaluate(X_scaled, y_true, y_bin, class_names):
        y_pred = clf.predict(X_scaled).ravel()
        y_prob = clf.predict_proba(X_scaled)

        f1_macro = f1_score(y_true, y_pred, average="macro")
        roc_auc_macro = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
        pr_auc_macro = average_precision_score(y_bin, y_prob, average="macro")

        f1_per_class = f1_score(y_true, y_pred, labels=class_names, average=None)
        f1_per_class_dict = dict(zip(class_names, f1_per_class))

        cm = confusion_matrix(y_true, y_pred, labels=class_names)

        return {
            "f1_macro": f1_macro,
            "roc_auc_macro": roc_auc_macro,
            "pr_auc_macro": pr_auc_macro,
            "f1_per_class": f1_per_class_dict,
            "confusion_matrix": cm
        }

    return {
        "C": evaluate(X_eval_scaled, y_eval, y_eval_bin, classes),
        "D": evaluate(X_gen_scaled, y_gen, y_gen_bin, classes)
    }
    
def plot_confusion_matrix(conf_matrix, class_names, scores_dict, title="Confusion Matrix", save_path=None):
    f1 = scores_dict["f1_macro"]
    roc_auc = scores_dict["roc_auc_macro"]
    pr_auc = scores_dict["pr_auc_macro"]
    f1_per_class = scores_dict.get("f1_per_class", {})

    f1_column = np.array([[f1_per_class.get(cls, 0)] for cls in class_names], dtype=np.float32)
    conf_matrix_extended = np.hstack([conf_matrix, f1_column])

    extended_col_names = class_names + ["F1"]

    annotations = []
    for i in range(conf_matrix.shape[0]):
        row = []
        for j in range(conf_matrix.shape[1]):
            row.append(str(int(conf_matrix[i, j])))
        row.append(f"{f1_column[i, 0]:.3f}")
        annotations.append(row)

    plt.figure(figsize=(9, 6))
    sns.heatmap(conf_matrix_extended, annot=annotations, fmt="", cmap="Blues",
                xticklabels=extended_col_names, yticklabels=class_names, cbar=False)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{title}\nF1: {f1:.3f} | ROC AUC: {roc_auc:.3f} | PR AUC: {pr_auc:.3f}")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
results = eval(df_train, df_test, df_val)
plot_confusion_matrix(results["C"]["confusion_matrix"], ["ddos", "normal", "sus"], results["C"], title="Results on C", save_path="images/results_c.png")
plot_confusion_matrix(results["D"]["confusion_matrix"], ["ddos", "normal", "sus"], results["D"], title="Results on D", save_path="images/results_d.png")