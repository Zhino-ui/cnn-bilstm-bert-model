import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 1. Load CSV file
df = pd.read_csv('kdd_database1.csv')

# 2. Data cleaning
df.drop_duplicates(inplace=True)                  # Remove duplicate rows
df.dropna(inplace=True)                           # Remove rows with missing values
if 'num_outbound_cmds' in df.columns:
    df.drop(['num_outbound_cmds'], axis=1, inplace=True)  # Drop known non-informative column if present

# 3. Extract relevant text features for BERT input
text_columns = ['protocol_type', 'service', 'flag']
df[text_columns] = df[text_columns].astype(str)   # Ensure they're strings
texts = df[text_columns].agg(' '.join, axis=1).tolist()

# 4. Convert labels: normal = 0, all others = 1 (attack)
labels = df['label'].map({'normal': 0}).fillna(1).astype(int)

# 5. Split dataset: 70% train, 30% test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)

# 6. Tokenization
max_len = 128
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, max_length=max_len, padding='max_length', truncation=True, return_tensors='tf')
test_encodings = tokenizer(test_texts, max_length=max_len, padding='max_length', truncation=True, return_tensors='tf')

# 7. Model definition: BERT + CNN + BiLSTM
# bert_model = TFBertModel.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased', output_hidden_states=False)
input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
bert_outputs = bert_model(input_ids, attention_mask=attention_mask)
sequence_output = bert_outputs.last_hidden_state

cnn = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(sequence_output)
cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn)
bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(cnn)
dense = tf.keras.layers.Dense(32, activation='relu')(bilstm)
output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 8. Convert labels to tensors
train_labels_tensor = tf.constant(train_labels.values, dtype=tf.float32)
test_labels_tensor = tf.constant(test_labels.values, dtype=tf.float32)

# 9. Train the model
model.fit(
    {'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']},
    train_labels_tensor,
    epochs=5,
    batch_size=16,
    verbose=2
)

# 10. Evaluate on the test set
pred_probs = model.predict({'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']})
pred_labels = (pred_probs.flatten() >= 0.5).astype(int)
true_labels = test_labels.values

accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print("\nEvaluation Metrics on KDDCup99 (Test Set):")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
