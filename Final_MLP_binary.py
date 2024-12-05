import ast
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from formatAndSplit import get_data, prep_data, undersample, findBestWeight
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def check_class(target_class):
    print("\nCurrent class = ", target_class)
    print()
    TARGET_CLASS = target_class
    df = get_data()
    X_train, X_dev, X_test, y_train, y_dev, y_test = prep_data(df, TARGET_CLASS, 20)
    X_resampled, y_resampled = undersample(X_train, y_train, 1)

    input_shape = X_train.shape[1]

    checkpoint = ModelCheckpoint(
        f"MLP_model_for_{TARGET_CLASS}.h5", 
        monitor='accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),  # Input layer
        tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer 1
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    history = model.fit(
        X_resampled, y_resampled,
        validation_data=(X_dev, y_dev),
        epochs=80,
        batch_size=32,
        callbacks=[checkpoint], 
        verbose=0
    )



    model = load_model(f"MLP_model_for_{TARGET_CLASS}.h5")
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)

    y_pred_probs = model.predict(X_test)
    # Convert probabilities to binary outputs (0 or 1)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

for class_name in ["CHM2210", "STA2023", "MAC1147", "PSY2012", "BSC2010", "ECO2013", "ACG2021", "MAC2311", "ACG3103"]:
    check_class(class_name)
    