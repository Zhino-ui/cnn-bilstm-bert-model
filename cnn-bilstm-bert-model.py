import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Loading and preprocessing
print("Loading dataset...")

def load_data(normal_file, sql_file):
    with open(normal_file, 'r', encoding='utf-8') as f:
        normal_queries = [line.strip() for line in f]
    
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_queries = [line.strip() for line in f]
    
    # Creating DataFrame
    df_normal = pd.DataFrame({'text': normal_queries, 'label': 0})
    df_sql = pd.DataFrame({'text': sql_queries, 'label': 1})
    
    return pd.concat([df_normal, df_sql], ignore_index=True)

# Loading training data
df = load_data('./data/train_normal.txt', './data/train_sql.txt')

# Splitting data (80% train, 20% test)
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# 2. BERT Tokenization
print("Tokenizing data...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_encode(texts, max_len=128):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
    
    return np.squeeze(np.array(input_ids)), np.squeeze(np.array(attention_masks))

max_len = 128
X_train_ids, X_train_masks = bert_encode(X_train, max_len)
X_test_ids, X_test_masks = bert_encode(X_test, max_len)

# 3. Model Architecture
print("Building model...")

# BERT base model
bert = TFBertModel.from_pretrained('bert-base-uncased')

# Input layers
input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32)
attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32)

# BERT embeddings
bert_output = bert(input_ids, attention_mask=attention_mask)[0]

# CNN-BiLSTM layers
x = tf.keras.layers.Conv1D(64, 3, activation='relu')(bert_output)
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)

# Classification head
x = tf.keras.layers.Dense(32, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(2e-5),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# 4. Class Weighting
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# 5. Training
print("Training model...")
history = model.fit(
    [X_train_ids, X_train_masks],
    y_train,
    validation_split=0.1,
    epochs=3,
    batch_size=16,
    class_weight=class_weights
)

# 6. Evaluation
print("Evaluating model...")
y_pred = (model.predict([X_test_ids, X_test_masks]) > 0.5).astype(int)

# Generate metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1-Score: {f1_score(y_test, y_pred)}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save model
model.save('sqli_detector.h5')
print("Model saved as sqli_detector.h5")
