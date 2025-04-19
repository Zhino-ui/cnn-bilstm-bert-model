import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 1. Small SQLi dataset (6 SQLi, 6 benign)
texts = [
    # SQLi examples
    "admin' OR 1=1--",
    "' OR ''='",
    "105 OR 1=1",
    "1; DROP TABLE users",
    "'; EXEC xp_cmdshell('dir');--",
    "105; SELECT * FROM users WHERE 'a'='a'",
    # Benign examples
    "SELECT * FROM users WHERE id = 2",
    "INSERT INTO customers (name, city) VALUES ('John', 'London')",
    "UPDATE products SET price = 10 WHERE id = 5",
    "DELETE FROM orders WHERE order_id = 100",
    "SELECT name FROM employees WHERE department = 'HR'",
    "SELECT * FROM login WHERE username = 'alice' AND password = '1234'"
]
labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 1=SQLi, 0=benign

# 2. Tokenization
max_len = 32
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(texts, max_length=max_len, padding='max_length', truncation=True, return_tensors='tf')

# 3. Model definition (BERT + CNN + BiLSTM)
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
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

# 4. Training (use all data for demonstration; in practice, use train/test split)
labels_tensor = tf.constant(labels, dtype=tf.float32)
model.fit(
    {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask']},
    labels_tensor,
    epochs=5,
    batch_size=2,
    verbose=2
)

# 5. Evaluation
pred_probs = model.predict({'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask']})
pred_labels = (pred_probs.flatten() >= 0.5).astype(int)
true_labels = np.array(labels)

accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print("\nEvaluation Metrics on Small SQLi Dataset:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
