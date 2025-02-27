import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, f1_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications import (ResNet152, ResNet50, DenseNet121, VGG16, EfficientNetB0, InceptionV3, MobileNetV2, InceptionResNetV2)
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inceptionv3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenetv2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, concatenate, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


busi_dataset_path = '/path/to/datasets'

# Loading Betti Vectors and Labels
df_betti = pd.read_csv("Disc_Edema_vs_Healthy.csv")
X_tda = df_betti.iloc[:, :-1].to_numpy()
y_tda = df_betti['Label'].to_numpy()

# Encoding  labels
y_combined = LabelEncoder().fit_transform(y_tda)
y_combined = tf.keras.utils.to_categorical(y_combined, 2)

# Loading image paths and labels
image_lists = {}
for folder_name in ["Healthy", "Disc_Edema"]:
    image_paths = []
    for root, dirs, files in os.walk(os.path.join(busi_dataset_path, folder_name)):
        for file in files:
            if file.endswith(('.pgm', '.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    image_lists[folder_name] = image_paths

all_images = image_lists['Healthy'] + image_lists['Disc_Edema']
labels = [0] * len(image_lists['Healthy']) + [1] * len(image_lists['Disc_Edema'])
df_image = pd.DataFrame({'image': all_images, 'Label': labels})

def calculate_sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  
    specificity = tn / (tn + fp)  
    return sensitivity, specificity

def create_preprocessing(model_name):
    preprocessors = {
        "DenseNet121": (DenseNet121, preprocess_densenet),
        "InceptionV3": (InceptionV3, preprocess_inceptionv3)
    }
    return preprocessors[model_name]

def preprocess_image(img_path, model_name):
    _, preprocess_function = create_preprocessing(model_name)
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized)
    preprocessed_img = preprocess_function(img_array)
    return preprocessed_img

def create_mlp(input_dim):
    input_mlp = Input(shape=(input_dim,))
    x = Dense(300, activation='relu')(input_mlp)
    x = Dense(150, activation='relu')(x)
    x = Dense(0, activation='relu')(x)
    return Model(inputs=input_mlp, outputs=x)

def create_cnn_mlp_model(model_name, tda_num_features, n_classes):
    base_model_class, preprocess_function = create_preprocessing(model_name)
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    for layer in base_model.layers:
        layer.trainable = True

    mlp = create_mlp(tda_num_features)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(base_model.output)
    x = MaxPooling2D(2, 2)(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    combined = concatenate([mlp.output, x])
    x = Dense(256, activation='relu')(combined)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(n_classes, activation='sigmoid')(x)

    model = Model(inputs=[mlp.input, base_model.input], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
    return model

# Preprocessing Images and Training them
model_names = [ "DenseNet121", "InceptionV3"]
results = []

for model_name in model_names:
    images_data = [preprocess_image(img_path, model_name) for img_path in df_image['image']]
    data_array_images = np.array(images_data)

    X_train_tda, X_test_tda, X_train_images, X_test_images, y_train, y_test = train_test_split(
        X_tda, data_array_images, y_combined, test_size=0.2, random_state=42, stratify=y_combined.argmax(axis=1))

    model = create_cnn_mlp_model(model_name, X_tda.shape[1], 2)
    model.fit(x=[X_train_tda, X_train_images], y=y_train, validation_data=([X_test_tda, X_test_images], y_test), epochs=100, batch_size=64)

    y_pred = model.predict([X_test_tda, X_test_images])
    y_pred_bin = (y_pred > 0.5).astype(int)
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred_bin, axis=1)

    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    balanced_acc = balanced_accuracy_score(y_test_labels, y_pred_labels)
    weighted_f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
    auc = roc_auc_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test_labels, y_pred_labels, average='micro')
    recall = recall_score(y_test_labels, y_pred_labels, average='micro')
    f1 = f1_score(y_test_labels, y_pred_labels)
    pr_auc = average_precision_score(y_test, y_pred)
    sensitivity, specificity = calculate_sensitivity_specificity(y_test_labels, y_pred_labels)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Weighted F1-Score": weighted_f1,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "PR AUC": pr_auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('Disc_Edema_CNN.csv', index=False)
print("Training complete. Results saved to Combined_Model_Results.csv.")

