# Airline Passenger Satisfatcion Classification
### This model tries to understand the services of an aviation company through a passenger satisfaction survey
# Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import pickle

# 1. Upload Datasets
df_test = pd.read_csv(os.path.join(Path(os.getcwd()).parents[0], 'data', 'test.csv'))
df_train = pd.read_csv(os.path.join(Path(os.getcwd()).parents[0], 'data', 'train.csv'))

## 1.1 Concatenating the Datasets
df_concat = pd.concat([df_train, df_test]).reset_index()
df_concat.shape

## 1.2 Renaming columns
df_rename = df_concat.copy()
df_concat.info()
df_rename.columns
new_columns = [
    new_name.replace(' ', '_').lower()
    for new_name in df_rename.columns
    ]
df_rename.columns = new_columns

### 1.2.1 Initial data cleaning
drop_cols = ['index', 'unnamed:_0', 'id']
df_rename.drop(drop_cols, axis = 1, inplace = True)

#### A: There's no duplicate ids. So, there's no reason to maintain this attribute
# 2. Data exploratory analysis
df_exploratory = df_rename.copy()

# 3. Dropping some categorical attributes
df_drop = df_exploratory.copy()
df_drop.head()
df_drop.drop(columns = 
    [   
        'gender',
        'age',
        'type_of_travel',
        'flight_distance',
        'departure_delay_in_minutes',
        'arrival_delay_in_minutes',
        'max_points'
    ]
    , inplace = True
)
df_drop.loc[3]
# 4. Encoding Categorical Data
df_encoder = df_drop.copy()

from sklearn.preprocessing import LabelEncoder
encoding_cols = df_encoder.select_dtypes(include='object').columns
encoding_cols
df_encoder['class'] = df_encoder.iloc[:, 1].apply(lambda cl: cl.replace(' ', '_'))
df_encoder.iloc[:, 1].unique()
le = LabelEncoder()
df_encoder['customer_type'] = le.fit_transform(df_encoder['customer_type'])
df_encoder['satisfaction'] = le.fit_transform(df_encoder['satisfaction'])
df_encoder['class'] = le.fit_transform(df_encoder['class'])
#### CLASS >> 0 = Business; 1 = Eco; 2 = Eco_Plus
#### CUSTOMER TYPE >> 0 = Loyal Customer; 1 = Disloyal Customer
#### SATISFACTION >> 0 = Neutral or Dissatisfied; 1 = Satisfied

# 5. Train & Test Split
from sklearn.model_selection import train_test_split

df_split = df_encoder.copy()
X = df_split.iloc[:, :-1].values
y = df_split.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# 6. Data Reduction
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, train_size=0.05, random_state=0)
sss.get_n_splits(X_train, y_train)

X = X_train
y = y_train

for train_index, test_index in sss.split(X, y):
    X_new_train = X[train_index]
    y_new_train = y[train_index]
X_new_train.shape, y_new_train.shape

# 7. The model: MLP Classifier
import pprint
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
parameters={
    'learning_rate': ["constant", "invscaling", "adaptive"], 
    'hidden_layer_sizes': [(10,), (10, 5)],
    'alpha': [0.0001, 0.001, 0.00001],
    'activation': ["logistic", "relu", "tanh"],
    'learning_rate_init': [0.001, 0.0001, 0.00001]    
            }

mlp = MLPClassifier()

clf = GridSearchCV(estimator = mlp, param_grid = parameters, n_jobs = -1, verbose = 3, cv = 10)
clf.fit(X_new_train, y_new_train)
clf.best_params_
mlp = MLPClassifier(activation='logistic', alpha=0.001, hidden_layer_sizes=(10, 5), learning_rate='adaptive', max_iter = 10000, verbose = 3, early_stopping = True)
mlp.fit(X_train, y_train)
pred_mlp = mlp.predict(X_test)
pred_mlp
conf_matrix = confusion_matrix(y_test, pred_mlp)
acc_score = accuracy_score(y_test, pred_mlp)
conf_matrix, acc_score
class_repo = classification_report(y_test, pred_mlp)
pprint.pprint(class_repo)
prob = mlp.predict_proba(X_test)
prob = prob[:, 1]
fper, tper, thresholds = roc_curve(y_test, prob)
plt.figure(figsize=(20, 10))
plt.plot(fper, tper, color='red', label='ROC')
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.show()


Pkl_Filename = os.path.join('model', 'Pickle_MLP_Model.pkl')

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(clf, file)
