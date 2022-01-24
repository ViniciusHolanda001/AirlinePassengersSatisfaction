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

# Pandas Set Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# 1. Upload Datasets
df_test = pd.read_csv(os.path.join(Path(os.getcwd()).parents[0], 'data', 'test.csv'))
df_train = pd.read_csv(os.path.join(Path(os.getcwd()).parents[0], 'data', 'train.csv'))
fig, ax = plt.subplots(figsize =(20, 10))

test = ax.bar('test_rows', df_test.shape[0], label='Test')
test_col = ax.bar('test_col', df_test.shape[1], label='Test_col')

train = ax.bar('train_rows', df_train.shape[0], label='Train')
train_col = ax.bar('train_col', df_train.shape[1], label='Train_col')

ax.bar_label(test, label_type='edge')
ax.bar_label(train, label_type='edge')

ax.bar_label(test_col, label_type='edge')
ax.bar_label(train_col, label_type='edge')
plt.ylim(0, 150000)
plt.legend()

plt.show()
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
len(df_rename['id'].unique())
drop_cols = ['index', 'unnamed:_0', 'id']
df_rename.drop(drop_cols, axis = 1, inplace = True)
#### A: There's no duplicate ids. So, there's no reason to maintain this attribute
# 2. Data exploratory analysis
df_exploratory = df_rename.copy()
#### Q.1: Is any null value?
df_exploratory.columns[df_exploratory.isnull().any()]
df_exploratory.isnull().sum()
#### A.1: Only one attribute have 393 null objects. That represents 0.3% of the data.
# Descriptive Data Analysis
df_exploratory.info()
# Count of dtypes
df_exploratory.dtypes.value_counts()
# A small sample
df_exploratory.sample(n = 5, random_state = 42)
# Memory usage
print(df_exploratory.memory_usage()[2] / 8, 'Bytes')
# Some statistical numbers
df_exploratory.describe(include = 'all').T
# Dimensions
df_exploratory.ndim
#### Age distribution
hist, ax = plt.subplots(figsize = (12, 8))

ax = sns.distplot(df_exploratory['age'], kde = True, color = 'black', norm_hist = True)
ax.set_title("Age Distribution")
plt.show()
#### Q.2: What is the proportion of male and female for Loyal Customer?
# Total of each gender
n_male = df_exploratory[df_exploratory['gender'] == 'Male'].count()[0]
n_female = df_exploratory[df_exploratory['gender'] == 'Female'].count()[0]

# Total of each gender for Loyal Customer
n_male_loyal = df_exploratory[(df_exploratory['gender'] == 'Male') & (df_exploratory['customer_type'] == 'Loyal Customer')].count()[0]
n_female_loyal = df_exploratory[(df_exploratory['gender'] == 'Female') & (df_exploratory['customer_type'] == 'Loyal Customer')].count()[0]
# Proportion of female that are loyal customers
proportion_fem = n_female_loyal / n_female * 100

# # Proportion of male that are loyal customers
proportion_male = n_male_loyal / n_male * 100

print(f'Loyal customers proportion: \n MALE: {proportion_male:.2f}% \n FEMALE: {proportion_fem:.2f}%')
#### A.2: Loyal customers proportion: 
        # MALE: 82.91%
        # FEMALE: 80.51%
#### Q.3: From the proportion of Loyal Customers what is the proportion of male and female that flew in Business class?
# Proportion of male that flew in the business class
n_male_loyal_business = df_exploratory[(df_exploratory['gender'] == 'Male')
                            & (df_exploratory['customer_type'] == 'Loyal Customer')
                            & (df_exploratory['class'] == 'Business')].count()[0]

# Proportion of female that flew in the business class
n_female_loyal_business = df_exploratory[(df_exploratory['gender'] == 'Female')
                            & (df_exploratory['customer_type'] == 'Loyal Customer')
                            & (df_exploratory['class'] == 'Business')].count()[0]
# Proportion of female that flew in the business class
proportion_fem_class = n_female_loyal_business / n_female_loyal * 100

# Proportion of male that flew in the business class
proportion_mal_class = n_male_loyal_business / n_male_loyal * 100

print(f'Loyal customers that flew in the business class proportion: \n MALE: {proportion_mal_class:.2f}% \n FEMALE: {proportion_fem_class:.2f}%')
#### A.3: Loyal customers that flew in the business class proportion: 
#         MALE: 49.84% 
#         FEMALE: 49.94%
#### Q.4: What is the proportion of customers that evaluated the company with an overall under 30 points?

points_customer_satis = df_exploratory[['gender', 'customer_type', 'age', 'type_of_travel', 'class', 'satisfaction']][(df_exploratory.iloc[ : , 6 : -3].sum(axis = 1) <= 30) & (df_exploratory['satisfaction'] == 'satisfied')].count()[0]             
points_customer_neut = df_exploratory[['gender', 'customer_type', 'age', 'type_of_travel', 'class', 'satisfaction']][(df_exploratory.iloc[ : , 6 : -3].sum(axis = 1) <= 30) & (df_exploratory['satisfaction'] == 'neutral or dissatisfied')].count()[0]

print(f'Only {points_customer_satis / points_customer_neut *100:.2f}% of the passengers who rated the company below 30 points classified with "satisfied"')
#### A.4: Only 15.05% of passengers who rated the company below 30 points classified it as "satisfied"

df_exploratory[['gender', 'customer_type', 'age', 'type_of_travel', 'class', 'satisfaction']] \
              [(df_exploratory.iloc[ : , 6 : -3]
              .sum(axis = 1) <= 15) & (df_exploratory['satisfaction'])]
#### Only two customers (loyal customers) that the overall points was less than 15 and yet classified with "satisfied"
df_exploratory[['gender', 'customer_type', 'age', 'type_of_travel', 'class', 'satisfaction']] \
              [(df_exploratory.iloc[ : , 6 : -3]
              .sum(axis = 1) < 30) & (df_exploratory['satisfaction'] == 'neutral or dissatisfied')].count()
null_satis = df_exploratory[(df_exploratory['arrival_delay_in_minutes'].isnull()) & (df_exploratory['satisfaction'] == 'satisfied')].count()[0]
null_total = df_exploratory[df_exploratory['arrival_delay_in_minutes'].isnull()].count()[0]

print(f'From null values the proportion of customers that classified with "satisfied" is: \n {null_satis / null_total * 100:.2f}%')
#### From null values the proportion of customers that classified with "satisfied" is: 
        # 42.24%
# Descriptive analysis
df_exploratory.groupby(['age', 'customer_type', 'class'])['satisfaction'].value_counts()
# Disloyal customers with age 31, 34, 53, 73, 74, 75 and 78 has no "satisfied" classifications
df_exploratory['age'].value_counts(sort = True, ascending = False)
plt.figure(figsize=(15, 10))
df_exploratory['age'].value_counts(normalize = True).plot(kind = 'bar')
plt.figure(figsize=(15, 10))
sns.scatterplot(data = df_exploratory,
                x = df_exploratory.age,
                y = df_exploratory.age.value_counts(),
                size = df_exploratory.age.value_counts(),
                alpha = 0.5,
                sizes=(20, 800),
                hue = 'satisfaction',
                )
plt.ylabel('Persons')
plt.show()
df_exploratory[(df_exploratory['age'] <= 18) & (df_exploratory['satisfaction'] == 'satisfied')].count()[0]
df_exploratory[(df_exploratory['age'] <= 18) & (df_exploratory['satisfaction'] != 'satisfied')].count()[0]
for atr in df_exploratory.columns:
    if df_exploratory[atr].dtype == "object":
        plt.figure(figsize=(10, 5))
        df_exploratory[atr].hist(bins = 5, grid = False, )
        plt.xlabel(atr)
        plt.show()
plt.figure(figsize=(20, 10))
sns.scatterplot(data = df_exploratory, x = 'class', y = 'age', hue = 'satisfaction')
plt.legend()
plt.show()
plt.figure(figsize=(20, 10))
sns.scatterplot(data = df_exploratory,
                x = 'departure_delay_in_minutes',
                y = 'arrival_delay_in_minutes',
                hue = 'satisfaction',
                style = 'satisfaction')
# This research has a metric of 0 to 5 for each one of the 15 attributes. Wich means the maximum result for each passenger is 75 points
df_exploratory['max_points'] = df_exploratory.iloc[ : , 6 : -4].sum(axis = 1)
print('Max = ', df_exploratory['max_points'].max(), '\nMin = ', df_exploratory['max_points'].min())
df_exploratory[df_exploratory['max_points'] >= 70]
df_exploratory[df_exploratory['max_points'] <= 15]
df_exploratory.groupby(['gender', 'customer_type', 'class'])['satisfaction'].value_counts()
fig, ax = plt.subplots(figsize = (20, 10))

X = df_exploratory[df_exploratory['customer_type'] == 'Loyal Customer'].count()
y = df_exploratory[(df_exploratory['customer_type'] == 'Loyal Customer') &
    (df_exploratory['satisfaction'] == 'satisfied')].count()

customer = ax.bar('customer_type', X, label='Loyal Customer')
satisfaction = ax.bar('satisfaction', y, label='satisfied')

ax.bar_label(customer, label_type='edge')
ax.bar_label(satisfaction, label_type='edge')

plt.ylim(0, 150000)
plt.legend()

plt.show()

df_exploratory.describe()
df_concat.plot(kind = 'scatter', x = 'Age', y = 'Food and drink', s = 'Age')
df_concat.hist(bins=50, figsize=(20, 15))
plt.show()
plt.figure(figsize=(20, 10))
sns.boxplot(data = df_concat, x = 'Age', y = 'Customer Type')
plt.show()
plt.figure(figsize=(20, 10))
sns.boxplot(data = df_concat['Flight Distance'])
plt.show()
plt.figure(figsize=(20, 10))
df_concat['Age'].hist()
plt.show()
plt.figure(figsize=(20, 10))
sns.scatterplot(data=df_exploratory,  x='departure_delay_in_minutes', y='arrival_delay_in_minutes', hue='satisfaction', style='satisfaction')
plt.show()
plt.figure(figsize=(20, 10))
sns.heatmap(df_exploratory.corr(), annot=True, center = 0.01, robust = True, linecolor = 'black', alpha = 0.9)
plt.show()
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
df_encoder.head()
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
