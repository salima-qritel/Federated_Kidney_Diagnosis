import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
import tensorflow as tf

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    X = df.drop("Diagnosis", axis=1).values
    y = df["Diagnosis"].values

    # Split initial
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Robust scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Détection des outliers avec IsolationForest
    iso = IsolationForest(contamination=0.05, random_state=42)
    is_inlier = iso.fit_predict(X_train_scaled) == 1

    X_train_filtered = X_train_scaled[is_inlier]
    y_train_filtered = y_train[is_inlier]

    # SMOTE après suppression des outliers
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_filtered, y_train_filtered)

    y_train_resampled = y_train_resampled.astype(np.float32)
    y_test = y_test.astype(np.float32)

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, scaler

def build_mlp(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )
    return model

def evaluate_tf_model(model, X_test, y_test):
    y_proba = model.predict(X_test).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    return acc, f1_macro, f1_weighted, precision, recall, roc
