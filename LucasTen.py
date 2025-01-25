import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras import layers
from keras import utils
from sklearn.model_selection import train_test_split

# Step 1: Load and Preprocess the Data
def load_data(file_path):
    data = pd.read_csv(file_path)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values
    X_top = X[:, :784].reshape(-1, 28, 28, 1) / 255.0
    X_mid = X[:, 784:1568].reshape(-1, 28, 28, 1) / 255.0
    X_bot = X[:, 1568:].reshape(-1, 28, 28, 1) / 255.0
    X_three = np.concatenate((X_top, X_mid, X_bot), axis=1)
    return X_top, X_mid, X_bot, y, X_three

def plotImg(x):
    img = x
    plt.imshow(img, cmap='gray')
    plt.show()
    return

# Step 2: Build the Odd/Even Classifier
def build_odd_even_classifier():
    model = Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Build the Digit Classifier
def build_digit_classifier():
    model = Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Train the Models
def train_models(train_file):
    X_top, X_mid, X_bot, y, X_three = load_data(train_file)

    y_odd_even = (y % 2 == 1).astype(int)

    X_top_train, X_top_val, y_oe_train, y_oe_val = train_test_split(X_top, y_odd_even, test_size=0.2)
    odd_even_model = build_odd_even_classifier()
    odd_even_model.fit(X_top_train, y_oe_train, validation_data=(X_top_val, y_oe_val), epochs=5)
    X_digits = np.concatenate((X_mid, X_bot), axis=0)
    y_digits = np.concatenate((y, y), axis=0)
    y_digits = utils.to_categorical(y_digits, num_classes=10)

    X_train, X_val, y_train, y_val = train_test_split(X_digits, y_digits, test_size=0.2)
    digit_model = build_digit_classifier()
    digit_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5)

    return odd_even_model, digit_model

# Step 5: Prediction
def predict_pipeline(odd_even_model, digit_model, X_top, X_mid, X_bot):
    is_odd = odd_even_model.predict(X_top).flatten() > 0.5
    predictions = []
    for i in range(len(X_top)):
        if is_odd[i]:
            predictions.append(np.argmax(digit_model.predict(X_mid[i:i+1])))
        else:
            predictions.append(np.argmax(digit_model.predict(X_bot[i:i+1])))
    return np.array(predictions)