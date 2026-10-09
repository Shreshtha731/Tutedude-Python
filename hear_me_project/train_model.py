import json
import os
import sys
from config import (
    ACTIONS_PATH,
    DATA_PATH,
    FEATURES_PER_FRAME,
    MODEL_PATH,
    SEQUENCE_LENGTH,
)
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def augment_sequence(seq):
  """Synthetically expands one sample into 6 robust variations."""
  variants = [seq]
  # Jitter noise
  variants.append(seq + np.random.normal(0, 0.012, seq.shape))
  # Distance scale invariant perturbations
  variants.append(seq * 1.05)
  variants.append(seq * 0.95)
  # Temporal window phase shifts
  variants.append(np.roll(seq, shift=2, axis=0))
  variants.append(np.roll(seq, shift=-2, axis=0))
  return variants


def main():
  if not os.path.exists(DATA_PATH):
    print(f"[!] Error: {DATA_PATH} directory not found.")
    sys.exit(1)

  actions = sorted(
      [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
  )
  if len(actions) == 0:
    print(f"[!] No class subfolders found inside {DATA_PATH}.")
    sys.exit(1)

  print(f"[*] Detected Action Classes ({len(actions)}): {actions}")
  label_map = {action: idx for idx, action in enumerate(actions)}

  sequences, labels = [], []

  for action in actions:
    action_folder = os.path.join(DATA_PATH, action)
    takes = [
        t
        for t in os.listdir(action_folder)
        if os.path.isdir(os.path.join(action_folder, t))
    ]

    for take in takes:
      seq_window = []
      take_dir = os.path.join(action_folder, take)

      for f_idx in range(SEQUENCE_LENGTH):
        file_path = os.path.join(take_dir, f"{f_idx}.npy")
        if os.path.exists(file_path):
          feat = np.load(file_path)
          if len(feat) < FEATURES_PER_FRAME:
            feat = np.pad(
                feat, (0, FEATURES_PER_FRAME - len(feat)), mode="constant"
            )
          seq_window.append(feat[:FEATURES_PER_FRAME])
        else:
          seq_window.append(np.zeros(FEATURES_PER_FRAME, dtype=np.float32))

      raw_seq = np.array(seq_window, dtype=np.float32)
      for variant in augment_sequence(raw_seq):
        sequences.append(variant)
        labels.append(label_map[action])

  X = np.array(sequences, dtype=np.float32)
  y = to_categorical(labels, num_classes=len(actions)).astype(np.float32)

  X_train, X_val, y_train, y_val = train_test_split(
      X, y, test_size=0.15, random_state=42, shuffle=True
  )

  print(f"[+] Loaded {len(sequences)} total augmented samples.")

  # Dual-Layer Bi-LSTM Architecture matching Report Chapter 5.2[cite: 1, 16, 18]
  model = Sequential([
      Bidirectional(
          LSTM(64, return_sequences=True),
          input_shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME),
      ),  #[cite: 1, 18]
      Dropout(0.2),  #[cite: 1, 18]
      Bidirectional(LSTM(128, return_sequences=False)),  #[cite: 1, 18]
      Dropout(0.3),  #[cite: 1, 18]
      Dense(64, activation="relu"),  #[cite: 1, 18]
      Dense(len(actions), activation="softmax"),  #[cite: 1, 18]
  ])

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  #[cite: 18]
      loss="categorical_crossentropy",  #[cite: 18]
      metrics=["accuracy"],  #[cite: 18]
  )

  callbacks = [
      EarlyStopping(
          monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
      ),
      ModelCheckpoint(filepath=MODEL_PATH, monitor="val_accuracy", save_best_only=True),
  ]

  print("\n[*] Training network...")
  history = model.fit(
      X_train,
      y_train,
      validation_data=(X_val, y_val),
      epochs=40,
      batch_size=16,
      callbacks=callbacks,
      verbose=1,
  )

  model.save(MODEL_PATH)
  with open(ACTIONS_PATH, "w") as f:
    json.dump(actions, f)

  print("\n" + "=" * 50)
  print(f"[✓] Artifacts saved: {MODEL_PATH} and {ACTIONS_PATH}")
  print(
      f"[✓] Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%"
  )
  print("=" * 50)


if __name__ == "__main__":
  main()