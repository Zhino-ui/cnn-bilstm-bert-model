import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

# 1. Loadingi the CSIC 2010 dataset
df = pd.read_csv('csic_database.csv')
print("Columns in the dataset:", df.columns)

# Converting labels to binary format: 0 for benign, 1 for SQLi
print('converting...')
df['label'] = df['classification'].astype(int)
print(df['classification'].unique())

# 2. Cleaning URLs removed unwanted info
print('cleaning...')
def clean_url(text):
    text = text.lower()                                
    text = re.sub(r'\d+', 'NUM', text)                
    text = re.sub(r'[\W_]+', ' ', text)                
    text = re.sub(r'\s+', ' ', text).strip()          
    return text

texts = df['URL'].astype(str).apply(clean_url).tolist()
labels = df['label'].tolist()

# 3. Tokenization
print('tokenizing')
max_len = 128
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(
    texts,
    max_length=max_len,
    padding='max_length',
    truncation=True,
    return_tensors='tf'
)

# 4. Spliting the data 
('spliting data...')
X_train_ids, X_test_ids, y_train, y_test = train_test_split(
    encodings['input_ids'].numpy(),
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)
X_train_masks, X_test_masks = train_test_split(
    encodings['attention_mask'].numpy(),
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# 5.the model (BERT + CNN + BiLSTM)
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
bert_outputs = bert_model(input_ids, attention_mask=attention_mask)
sequence_output = bert_outputs.last_hidden_state

# CNN + BiLSTM layers
cnn = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(sequence_output)
cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn)
bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(cnn)
dense = tf.keras.layers.Dense(32, activation='relu')(bilstm)
output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 6. Training the model
print("training the model...")
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)  # Converting to TensorFlow tensor
model.fit(
    {'input_ids': X_train_ids, 'attention_mask': X_train_masks},
    y_train_tensor,
    epochs=1,                      
    batch_size=8,                 
    verbose=1
)

# 7. Evaluating the model
print("evaluating the model ...")
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)  # Converting to TensorFlow tensor
pred_probs = model.predict({'input_ids': X_test_ids, 'attention_mask': X_test_masks})
pred_labels = (pred_probs.flatten() >= 0.5).astype(int)

unique, counts = np.unique(pred_labels, return_counts=True)
print("Prediction counts:", dict(zip(unique, counts)))

# Evaluation Metrics
accuracy = accuracy_score(y_test, pred_labels)
precision = precision_score(y_test, pred_labels)
recall = recall_score(y_test, pred_labels)
f1 = f1_score(y_test, pred_labels)

print("\nEvaluation Metrics on CSIC 2010 Dataset:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(confusion_matrix(y_test, pred_labels))
