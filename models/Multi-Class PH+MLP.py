import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, 
                             balanced_accuracy_score, roc_auc_score)

batch_size = 128
num_classes = 8

data = pd.read_csv('/path/to/betti/vectors')

X = data.drop(columns=['label'], axis=1)
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_cat))

train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

test_dataset = test_dataset.batch(batch_size)

def lr_schedule(epoch):
    lr = 0.0001
    if epoch < 50:
        return lr
    elif epoch < 75:
        return lr * 0.1
    else:
        return lr * 0.1 * 0.1  

lr_scheduler = LearningRateScheduler(lr_schedule)

model = Sequential([
    Input(shape=(400,)),    

    Dense(256, activation='relu'),
    Dropout(0.1),

    Dense(256, activation='relu'),
    Dropout(0.1),

    Dense(128, activation='relu'),
    Dropout(0.1),

    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')]
             )

# train model
history = model.fit(train_dataset, 
                    epochs=100,
                    callbacks=[lr_scheduler],
                    )

pred_probs = model.predict(X_test)  
pred_labels = np.argmax(pred_probs, axis=1)  

true_labels = y_test

# compute metrics 
acc = accuracy_score(true_labels, pred_labels)
weighted_f1 = f1_score(true_labels, pred_labels, average='weighted')
mcc = matthews_corrcoef(true_labels, pred_labels)
balanced_acc = balanced_accuracy_score(true_labels, pred_labels)
roc_auc = roc_auc_score(true_labels, pred_probs, multi_class='ovr')

print("\nTest Metrics:")
print(f"Test ROC-AUC:           {roc_auc:.4f}")
print(f"Test Accuracy:          {acc:.4f}")
print(f"Test Weighted F1:       {weighted_f1:.4f}")
print(f"Test MCC:               {mcc:.4f}")
print(f"Test Balanced Accuracy: {balanced_acc:.4f}")