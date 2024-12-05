import ast
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from formatAndSplit_multiclass import get_data, prep_data
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

def check_class(target_class):
    print("\nCurrent class = ", target_class)
    df = get_data()
    X_train, X_dev, X_test, y_train, y_dev, y_test = prep_data(df, target_class, 20)

    input_shape = X_train.shape[1]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),  # Input layer
        tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer 1
        tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer 2
        tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer 2
        tf.keras.layers.Dense(4, activation='softmax')  # Output layer for 4-class classification
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy'])

    checkpoint = ModelCheckpoint(f"MLP_model_for_{target_class}.h5", 
                                monitor='val_categorical_accuracy',  # Monitor validation accuracy
                                save_best_only=True, 
                                mode='max', 
                                verbose=0)

    # Train the model with the checkpoint callback
    history = model.fit(X_train, y_train,
                        validation_data=(X_dev, y_dev),
                        epochs=70, 
                        batch_size=32, 
                        callbacks=[checkpoint],
                        verbose=0)




    model = load_model(f"MLP_model_for_{target_class}.h5")
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Predict class probabilities
    y_pred_probs = model.predict(X_test)

    # Convert probabilities to predicted class labels (choose the class with highest probability)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Convert one-hot encoded y_dev to class labels for comparison
    y_test_labels = np.argmax(y_test, axis=1)

    # Print classification report for multi-class classification
    print(classification_report(y_test_labels, y_pred, target_names=["A class", "B class", "C class", "Fail"]))

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test_labels, y_pred)
    print(conf_matrix)

for class_name in ["CHM2210", "STA2023", "MAC1147", "PSY2012", "BSC2010", "ECO2013", "ACG2021", "MAC2311", "ACG3103"]:
    check_class(class_name)
    