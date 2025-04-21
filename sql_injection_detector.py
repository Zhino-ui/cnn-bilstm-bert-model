import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight

# 1. Loading the CSIC 2010 dataset
print("Loading data...")
df = pd.read_csv('csic_database.csv')
print(df.columns)
print(df.head())

# Using existing binary classification labels and URL column of the dataset
df['label'] = df['classification']
X = df['URL'].astype(str)
y = df['label'].astype(int)

# Splitting data
print("Splitting the data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Tokenization with BERT
print("Tokenizing...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_encode(texts, tokenizer, max_len=128):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.vstack(input_ids), np.vstack(attention_masks)

max_len = 128
X_train_ids, X_train_masks = bert_encode(X_train, tokenizer, max_len)
X_test_ids, X_test_masks = bert_encode(X_test, tokenizer, max_len)

# 3. Building the CNN-BiLSTM-BERT model
print("Building the model...")
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')

bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]  # (batch, seq_len, hidden)

# CNN layer
cnn = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(bert_output)
cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn)

# BiLSTM layer
bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(cnn)

# Dense layers
dense = tf.keras.layers.Dense(32, activation='relu')(bilstm)
output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

model.summary()

# 4.class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# 5. Training the model
print("Training the model...")
history = model.fit(
    [X_train_ids, X_train_masks],
    y_train,
    validation_split=0.1,
    epochs=3,
    batch_size=16,
    class_weight=class_weights
)

# 6. Evaluating the model
print("Evaluating performance...")
y_pred_probs = model.predict([X_test_ids, X_test_masks])
y_pred = (y_pred_probs > 0.5).astype(int)

print("Predicted class counts:", np.unique(y_pred, return_counts=True))
print("Actual class counts:", np.unique(y_test, return_counts=True))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Malicious'],
            yticklabels=['Actual Normal', 'Actual Malicious'])
plt.title('Confusion Matrix')
plt.show()

# Training progress
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training Progress')
plt.legend()
plt.show()

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print(f"\n{' Metric ':-^30}")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Malicious']))

print("\nClass distribution:\n", df['label'].value_counts())

# 7. Save the model
model.save('sql_injection_detector.h5')
print("Model saved as sql_injection_detector.h5")
