import pandas as pd
from sklearn.model_selection import train_test_split

# dataset 
df = pd.read_csv('csic_database.csv')  # file location

#preprocessing
df['label'] = df['traffic'].apply(lambda x: 1 if 'anomalous' in x else 0)
X = df['request']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
