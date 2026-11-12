import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)

class NNResult:
    def __init__(self, history, y_test, y_pred, classes, log_dir):
        self.history = history
        self.y_test = y_test
        self.y_pred = y_pred
        self.classes = classes
        self.log_dir = log_dir

    def accuracy(self):
        return float(accuracy_score(self.y_test, self.y_pred))

    def f1_macro(self):
        return float(f1_score(self.y_test, self.y_pred, average="macro", zero_division=0))

    def __str__(self):
        return (
            "Neural Network (TensorFlow / Keras)\n"
            f"  Test accuracy : {self.accuracy():.4f}\n"
            f"  Test F1 macro : {self.f1_macro():.4f}\n"
            f"  TensorBoard   : tensorboard --logdir {self.log_dir}\n"
        )


def build_nn_classifier(
    dataframe, target, test_ratio=0.2, random_state=42,
    epochs=100, batch_size=16, log_dir="logs/fit",
):
    import tensorflow as tf

    os.makedirs(log_dir, exist_ok=True)

    feature_columns = [
        c for c in dataframe.select_dtypes(include=["number"]).columns if c != target
    ]
    X = dataframe[feature_columns].values
    y = dataframe[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classes = np.unique(y)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train_idx = np.array([class_to_idx[c] for c in y_train])

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(len(classes), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_graph=True
    )
    history = model.fit(
        X_train, y_train_idx,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[tensorboard_cb],
        verbose=0,
    )

    y_pred_idx = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_pred = np.array([classes[i] for i in y_pred_idx])

    return NNResult(history, y_test, y_pred, classes, log_dir)