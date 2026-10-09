import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

DATA_PATH = os.path.join("MP_Data")
ACTIONS = np.array(
    [
        action
        for action in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, action))
    ]
)
SEQUENCE_LENGTH = 30

label_map = {label: num for num, label in enumerate(ACTIONS)}

sequences, labels = [], []
for action in ACTIONS:
    for sequence in os.listdir(os.path.join(DATA_PATH, action)):
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            res = np.load(
                os.path.join(
                    DATA_PATH, action, sequence, f"{frame_num}.npy"
                )
            )
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)  # Shape: (samples, 30, 126)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Bidirectional LSTM Topology (30 frames, 126 features)
model = Sequential(
    [
        Bidirectional(
            LSTM(64, return_sequences=True), input_shape=(30, 126)
        ),
        Dropout(0.2),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(len(ACTIONS), activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ModelCheckpoint(
        "action_model.h5", monitor="val_categorical_accuracy", save_best_only=True
    ),
]

model.fit(
    X_train,
    y_train,
    epochs=120,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
)

model.save("action_model.h5")
np.save("actions.npy", ACTIONS)
print("Model training complete and saved as action_model.h5")   