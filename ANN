import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset from Git
st.title("Customer Churn Prediction using ANN")

git_file_path = "Telco_customer_churn.xlsx"
df = pd.read_excel(git_file_path)
st.write("### Dataset Overview")
st.dataframe(df.head())

# Preprocessing
categorical_cols = ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop(columns=['Churn Label', 'Churn Value', 'Churn Score', 'CLTV', 'Churn Reason'])
y = df['Churn Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Sidebar for hyperparameter selection
st.sidebar.header("Hyperparameter Selection")
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32, 16)
dense_layers = st.sidebar.radio("Number of Dense Layers", [3, 4])

# Build ANN model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
for _ in range(dense_layers):
    model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=batch_size, verbose=0)

# Plot Loss Graph
st.write("### Loss Graph")
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='Train Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()
st.pyplot(fig)

# Plot Accuracy Graph
st.write("### Accuracy Graph")
fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='Train Accuracy')
ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
ax.legend()
st.pyplot(fig)
