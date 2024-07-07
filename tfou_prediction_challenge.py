#!/usr/bin/env python
# coding: utf-8

# # Togo Fiber Optics Uptake Prediction Challenge
# 
# ##### Households and businesses fiber optics access prediction

# In[1]:


# Importing the necessary modules
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
classification_report, confusion_matrix, accuracy_score, 
precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,  MinMaxScaler
import seaborn as sns

from pprint import pprint


# ### Loading Data

# In[2]:


# Train and test dataset reading
df_train = pd.read_csv("datasets/Train.csv")
df_test = pd.read_csv("datasets/Test.csv")


# ### Exploring and processing the Data

# In[3]:


# Shape of train and test datasets
print("Shape of training dataset :", df_train.shape)
print("Shape of testing dataset :", df_test.shape)


# In[4]:


# Counting all the missing values of train and test datasets
print("Number of NA data of training dataset :", df_train.isnull().sum().sum())
print("Number of NA data of testing dataset :", df_test.isnull().sum().sum())


# In[5]:


# Making a copy of train and test dataset
df_train_clean = df_train.copy()
df_test_clean = df_test.copy()


# In[6]:


# Computing modal values pour every columns
modes_train = df_train_clean.mode().iloc[0]
modes_test = df_test_clean.mode().iloc[0]


# In[7]:


# Remplace NaNs by the modal value of every column
df_train_clean = df_train_clean.fillna(modes_train)
df_test_clean = df_test_clean.fillna(modes_test)


# In[8]:


df_train_clean['Accès internet'].value_counts()


# In[9]:


# Checking if missing values exist again in train and test dataset
if df_train_clean.isnull().sum().sum() == 0:
    print("There is no more missing values in train and test dataset :")
    print("Number of NA data of training dataset :", df_train_clean.isnull().sum().sum())
    print("Number of NA data of testing dataset :", df_test_clean.isnull().sum().sum())
else:
    print("Process missing data before continue")


# In[10]:


# Counting number of duplicated data
if ((df_train_clean.duplicated().sum() == 0) and (df_test_clean.duplicated().sum() == 0)):
    print("There is no duplicated observation in training and testing dataset")
    print("Number of duplicating observatiion of training dataset :", df_train.duplicated().sum())
    print("Number of duplicating observatiion of testing dataset :", df_test.duplicated().sum())
else:
    print("Process duplicated data before continue")


# In[11]:


pprint(df_train_clean.info())

print("\n----------------------------------------------------------------------------- \n")

pprint(df_test_clean.info())


# *Train dataset : There are 4 043 columns, 4 002 numerical and 41 categorical variables.*
# 
# *Test dataset : There are 4 042 columns, 4 001 numerical and 41 categorical variables.*

# In[12]:


df_train_clean.head()


# In[13]:


df_test_clean.head()


# In[14]:


# Rename column name "Accès internet" to "acces_internet"
df_train2 = df_train_clean.rename(columns={'Accès internet': 'Target'})
df_test2 = df_test_clean


# In[15]:


numerical_columns = df_train2.select_dtypes(include=[np.number])

print("Numerical columns shape : ", numerical_columns.shape)

for col in numerical_columns.columns:
    print(col)


# In[16]:


# Removing dots from column names
df_train2.columns = df_train2.columns.str.replace('.', '')
df_test2.columns = df_test2.columns.str.replace('.', '')

# Checking if dots are removed
numerical_columns = df_train2.select_dtypes(include=[np.number])

for col in numerical_columns.columns:
    print(col)


# In[17]:


# List of categorical column names
categorical_columns = df_train2.select_dtypes(include='O')

print(categorical_columns.shape)

for col in categorical_columns:
    print(col)


# In[18]:


for col in categorical_columns:
  print(f"Modalités de la variable '{col}':")
  print(df_train2[col].value_counts())
  print("-----------------------------------------------")


# #### Rename some value of some categorical variable
# Certaines valeurs de certaines variables catégorielles ne sont pas lisibles ni compréhensible. Il est donc nécessaire de les renommer :
# 
#     - Pour la colonne 'TypeLogmt_1' : 
#         - 'Logement moderne' : 'Moderne'
#         - 'Logement semi-moderne' : 'SemiModerne'
#         - 'Logement traditionnel' : 'Traditionnel'.
#     
#     - Pour la colonne 'TypeLogmt_2' :
#         - 'Plusieurs logement' : 'Plusieurs'
#         - 'Logement unique' : Unique
#     
#     - Concernant la colonne 'TypeLogmt_3' :
#         - 'Logement � un niveau (plain-pied)' : 'UnNiveau'
#         - 'Logement ? un niveau (plain-pied)' : 'UnNiveau'
#         - 'Logement � plusieurs niveaux (� �tage)' : 'PlusieursNiveaux'
#         - 'Logement ? plusieurs niveaux (? ?tage)' : 'PlusieursNiveaux'
# 
#     - Pour la variable 'H08_Impute' :
#         - 'Electricit� (CEET) compteur dans la concession' : 'CeetCompteurDansLaConcession'
#         - 'Electricit? (CEET) compteur dans la concession' : 'CeetCompteurDansLaConcession'
#         - 'Electricit? (CEET) compteur hors concession' : 'CeetCompteurHorsConcession'
#         - 'Electricit� (CEET) compteur hors concession' : 'CeetCompteurHorsConcession'
#         - 'Torche/bougie' :'TorcheOuBougie'
#         - 'C?blage d??lectricit? du voisinnage' : 'ElectricitDuVoisinnage'
#         - 'C�blage d��lectricit� du voisinnage' : 'ElectricitDuVoisinnage'
#         - 'Autre (? pr?ciser)' : 'Autre'
#         - 'Autre (� pr�ciser)' : 'Autre'
#         - 'Energie solaire' : 'EnergieSolaire'
#         - 'Lampe ? p?trole' : 'LampeAPetrole'
#         - 'Lampe � p�trole' : 'LampeAPetrole'
#         - 'Groupe �lectrog�ne' : 'GroupeElectrogene'
#         - 'Groupe ?lectrog?ne' : 'GroupeElectrogene'
#         - 'Lampe ? gaz' : 'LampeAGaz'
# 
#     - Pour la variable 'H09_Impute' :
#         - 'Charbon de bois' : 'CharbonDeBois'
#         - 'Bois de chauffe' : 'BoisDeChauffe'
#         - '�lectricit�' : 'Electricite'
#         - '?lectricit?' : 'Electricite'  
#         - 'Ne cuisine pas' : 'NeCuisinePas'  
#         - 'P?trole' : 'Petrole'  
#         - 'P�trole' : 'Petrole'  
#         - 'Autre (? pr?ciser)' : 'Autre'  
#         - 'Autre (� pr�ciser' : 'Autre'  
#         - 'R�sidus v�g�taux/sciure de bois' : 'ResidusVegetauxOuSciureDeBois'  
#         - 'RR?sidus v?g?taux/sciure de boi' : 'ResidusVegetauxOuSciureDeBois'

# ###### Rename columns values on train and test set

# In[19]:


# rename values of TypeLogemt_1 variable
df_train2.loc[df_train2['TypeLogmt_1'] == 'Logement moderne', 'TypeLogmt_1'] = 'Moderne'
df_train2.loc[df_train2['TypeLogmt_1'] == 'Logement semi-moderne', 'TypeLogmt_1'] = 'SemiModerne'
df_train2.loc[df_train2['TypeLogmt_1'] == 'Logement traditionnel', 'TypeLogmt_1'] = 'Traditionnel'

# rename values of TypeLogemt_2 variable
df_train2.loc[df_train2['TypeLogmt_2'] == 'Plusieurs logement', 'TypeLogmt_2'] = 'Plusieurs'
df_train2.loc[df_train2['TypeLogmt_2'] == 'Logement unique', 'TypeLogmt_2'] = 'Unique'

# rename values of TypeLogemt_3 variable
df_train2.loc[df_train2['TypeLogmt_3'] == 'Logement � un niveau (plain-pied)', 'TypeLogmt_3'] = 'UnNiveau'
df_train2.loc[df_train2['TypeLogmt_3'] == 'Logement ? un niveau (plain-pied)', 'TypeLogmt_3'] = 'UnNiveau'
df_train2.loc[df_train2['TypeLogmt_3'] == 'Logement � plusieurs niveaux (� �tage)', 'TypeLogmt_3'] = 'PlusieursNiveaux'
df_train2.loc[df_train2['TypeLogmt_3'] == 'Logement ? plusieurs niveaux (? ?tage)', 'TypeLogmt_3'] = 'PlusieursNiveaux'

# rename values of H08_Impute variable
df_train2.loc[df_train2['H08_Impute'] == 'Electricit� (CEET) compteur dans la concession', 'H08_Impute'] = 'CeetCompteurDansLaConcession'
df_train2.loc[df_train2['H08_Impute'] == 'Electricit? (CEET) compteur dans la concession', 'H08_Impute'] = 'CeetCompteurDansLaConcession'

df_train2.loc[df_train2['H08_Impute'] == 'Electricit? (CEET) compteur hors concession', 'H08_Impute'] = 'CeetCompteurHorsConcession'
df_train2.loc[df_train2['H08_Impute'] == 'Electricit� (CEET) compteur hors concession', 'H08_Impute'] = 'CeetCompteurHorsConcession'

df_train2.loc[df_train2['H08_Impute'] == 'Torche/bougie', 'H08_Impute'] = 'TorcheOuBougie'

df_train2.loc[df_train2['H08_Impute'] == 'C?blage d??lectricit? du voisinnage', 'H08_Impute'] = 'ElectricitDuVoisinnage'
df_train2.loc[df_train2['H08_Impute'] == 'C�blage d��lectricit� du voisinnage', 'H08_Impute'] = 'ElectricitDuVoisinnage'

df_train2.loc[df_train2['H08_Impute'] == 'Autre (? pr?ciser)', 'H08_Impute'] = 'Autre'
df_train2.loc[df_train2['H08_Impute'] == 'Autre (� pr�ciser)', 'H08_Impute'] = 'Autre'

df_train2.loc[df_train2['H08_Impute'] == 'Energie solaire', 'H08_Impute'] = 'EnergieSolaire'
df_train2.loc[df_train2['H08_Impute'] == 'Lampe ? p?trole', 'H08_Impute'] = 'LampeAPetrole'
df_train2.loc[df_train2['H08_Impute'] == 'Lampe � p�trole', 'H08_Impute'] = 'LampeAPetrole'
df_train2.loc[df_train2['H08_Impute'] == 'Groupe �lectrog�ne', 'H08_Impute'] = 'GroupeElectrogene'
df_train2.loc[df_train2['H08_Impute'] == 'Groupe ?lectrog?ne', 'H08_Impute'] = 'GroupeElectrogene'
df_train2.loc[df_train2['H08_Impute'] == 'Lampe ? gaz', 'H08_Impute'] = 'LampeAGaz'

# rename values of H09_Impute variable
df_train2.loc[df_train2['H09_Impute'] == 'Charbon de bois', 'H09_Impute'] = 'CharbonDeBois'
df_train2.loc[df_train2['H09_Impute'] == 'Bois de chauffe', 'H09_Impute'] = 'BoisDeChauffe'
df_train2.loc[df_train2['H09_Impute'] == '�lectricit�', 'H09_Impute'] = 'Electricite'
df_train2.loc[df_train2['H09_Impute'] == '?lectricit?', 'H09_Impute'] = 'Electricite'
df_train2.loc[df_train2['H09_Impute'] == 'Ne cuisine pas', 'H09_Impute'] = 'NeCuisinePas'
df_train2.loc[df_train2['H09_Impute'] == 'P?trole', 'H09_Impute'] = 'Petrole'
df_train2.loc[df_train2['H09_Impute'] == 'P�trole', 'H09_Impute'] = 'Petrole'
df_train2.loc[df_train2['H09_Impute'] == 'Autre (? pr?ciser)', 'H09_Impute'] = 'Autre'
df_train2.loc[df_train2['H09_Impute'] == 'Autre (� pr�ciser)', 'H09_Impute'] = 'Autre'
df_train2.loc[df_train2['H09_Impute'] == 'R�sidus v�g�taux/sciure de bois', 'H09_Impute'] = 'ResidusVegetauxOuSciureDeBois'
df_train2.loc[df_train2['H09_Impute'] == 'R?sidus v?g?taux/sciure de bois', 'H09_Impute'] = 'ResidusVegetauxOuSciureDeBois'
#======================================


# In[20]:


# rename values of TypeLogemt_1 variable
df_test2.loc[df_test2['TypeLogmt_1'] == 'Logement moderne', 'TypeLogmt_1'] = 'Moderne'
df_test2.loc[df_test2['TypeLogmt_1'] == 'Logement semi-moderne', 'TypeLogmt_1'] = 'SemiModerne'
df_test2.loc[df_test2['TypeLogmt_1'] == 'Logement traditionnel', 'TypeLogmt_1'] = 'Traditionnel'

# rename values of TypeLogemt_2 variable
df_test2.loc[df_test2['TypeLogmt_2'] == 'Plusieurs logement', 'TypeLogmt_2'] = 'Plusieurs'
df_test2.loc[df_test2['TypeLogmt_2'] == 'Logement unique', 'TypeLogmt_2'] = 'Unique'

# rename values of TypeLogemt_3 variable
df_test2.loc[df_test2['TypeLogmt_3'] == 'Logement � un niveau (plain-pied)', 'TypeLogmt_3'] = 'UnNiveau'
df_test2.loc[df_test2['TypeLogmt_3'] == 'Logement ? un niveau (plain-pied)', 'TypeLogmt_3'] = 'UnNiveau'
df_test2.loc[df_test2['TypeLogmt_3'] == 'Logement � plusieurs niveaux (� �tage)', 'TypeLogmt_3'] = 'PlusieursNiveaux'
df_test2.loc[df_test2['TypeLogmt_3'] == 'Logement ? plusieurs niveaux (? ?tage)', 'TypeLogmt_3'] = 'PlusieursNiveaux'

# rename values of H08_Impute variable
df_test2.loc[df_test2['H08_Impute'] == 'Electricit� (CEET) compteur dans la concession', 'H08_Impute'] = 'CeetCompteurDansLaConcession'
df_test2.loc[df_test2['H08_Impute'] == 'Electricit? (CEET) compteur dans la concession', 'H08_Impute'] = 'CeetCompteurDansLaConcession'

df_test2.loc[df_test2['H08_Impute'] == 'Electricit? (CEET) compteur hors concession', 'H08_Impute'] = 'CeetCompteurHorsConcession'
df_test2.loc[df_test2['H08_Impute'] == 'Electricit� (CEET) compteur hors concession', 'H08_Impute'] = 'CeetCompteurHorsConcession'

df_test2.loc[df_test2['H08_Impute'] == 'Torche/bougie', 'H08_Impute'] = 'TorcheOuBougie'

df_test2.loc[df_test2['H08_Impute'] == 'C?blage d??lectricit? du voisinnage', 'H08_Impute'] = 'ElectricitDuVoisinnage'
df_test2.loc[df_test2['H08_Impute'] == 'C�blage d��lectricit� du voisinnage', 'H08_Impute'] = 'ElectricitDuVoisinnage'

df_test2.loc[df_test2['H08_Impute'] == 'Autre (? pr?ciser)', 'H08_Impute'] = 'Autre'
df_test2.loc[df_test2['H08_Impute'] == 'Autre (� pr�ciser)', 'H08_Impute'] = 'Autre'

df_test2.loc[df_test2['H08_Impute'] == 'Energie solaire', 'H08_Impute'] = 'EnergieSolaire'
df_test2.loc[df_test2['H08_Impute'] == 'Lampe ? p?trole', 'H08_Impute'] = 'LampeAPetrole'
df_test2.loc[df_test2['H08_Impute'] == 'Lampe � p�trole', 'H08_Impute'] = 'LampeAPetrole'
df_test2.loc[df_test2['H08_Impute'] == 'Groupe �lectrog�ne', 'H08_Impute'] = 'GroupeElectrogene'
df_test2.loc[df_test2['H08_Impute'] == 'Groupe ?lectrog?ne', 'H08_Impute'] = 'GroupeElectrogene'
df_test2.loc[df_test2['H08_Impute'] == 'Lampe ? gaz', 'H08_Impute'] = 'LampeAGaz'

# rename values of H09_Impute variable
df_test2.loc[df_test2['H09_Impute'] == 'Charbon de bois', 'H09_Impute'] = 'CharbonDeBois'
df_test2.loc[df_test2['H09_Impute'] == 'Bois de chauffe', 'H09_Impute'] = 'BoisDeChauffe'
df_test2.loc[df_test2['H09_Impute'] == '�lectricit�', 'H09_Impute'] = 'Electricite'
df_test2.loc[df_test2['H09_Impute'] == '?lectricit?', 'H09_Impute'] = 'Electricite'
df_test2.loc[df_test2['H09_Impute'] == 'Ne cuisine pas', 'H09_Impute'] = 'NeCuisinePas'
df_test2.loc[df_test2['H09_Impute'] == 'P?trole', 'H09_Impute'] = 'Petrole'
df_test2.loc[df_test2['H09_Impute'] == 'P�trole', 'H09_Impute'] = 'Petrole'
df_test2.loc[df_test2['H09_Impute'] == 'Autre (? pr?ciser)', 'H09_Impute'] = 'Autre'
df_test2.loc[df_test2['H09_Impute'] == 'Autre (� pr�ciser)', 'H09_Impute'] = 'Autre'
df_test2.loc[df_test2['H09_Impute'] == 'R�sidus v�g�taux/sciure de bois', 'H09_Impute'] = 'ResidusVegetauxOuSciureDeBois'
df_test2.loc[df_test2['H09_Impute'] == 'R?sidus v?g?taux/sciure de bois', 'H09_Impute'] = 'ResidusVegetauxOuSciureDeBois'
#======================================


# In[21]:


for col in categorical_columns:
  print(f"Modalités de la variable '{col}':")
  print(df_train2[col].value_counts())
  print("--------------------------------------------")


# In[22]:


sns.displot(df_train2['Target']);


# In[23]:


sns.displot(df_train2['TypeLogmt_1']);


# In[24]:


sns.displot(df_train2['TypeLogmt_2']);


# In[25]:


sns.displot(df_train2['TypeLogmt_3']);


# In[26]:


categorical_columns = categorical_columns.drop(columns = ['ID'], axis = 1)


# In[27]:


# Encoding categorical variables with LabelEncoder
label_encoder = LabelEncoder()
for col in categorical_columns:
    df_train2[col] = label_encoder.fit_transform(df_train2[col])
    df_test2[col] = label_encoder.fit_transform(df_test2[col])


# In[28]:


df_train2.head()


# In[29]:


df_test2.head()


# ##### Building and Training Dense Neural Network model

# In[30]:


y = df_train2.iloc[:,-1]
#Y = LabelEncoder().fit_transform(y)
X = df_train2.drop(columns = ['Target'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[31]:


id_train = X_train["ID"]
id_test = X_test["ID"]
id_df_test2 = df_test2['ID']

X_train = X_train.drop(columns = ["ID"], axis = 1)
X_test = X_test.drop(columns = ["ID"], axis = 1) 
df_test2 = df_test2.drop(columns = ["ID"], axis = 1)

df_test_val = df_test2.values

# Get the IDs of individuals from the 'ID' column
individual_ids = id_test


# In[32]:


# Normalization
# Initialize the StandardScaler
#scaler = StandardScaler()
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform both training and test data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

df_test_val = scaler.transform(df_test_val)


# ## Logistic Regression model

# In[33]:


# Define the Logistic Regression with l2 (Ridge) penality
log_model = LogisticRegression(penalty='l2', solver='newton-cg')

# Train the model
log_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_train_classes = log_model.predict(X_train)
y_pred_classes = log_model.predict(X_test)
y_pred_prob = log_model.predict_proba(X_test)[:, 1]
df_test2_pred_classes = log_model.predict(df_test_val)
df_test2_pred_prob = log_model.predict_proba(df_test_val)

# Evaluate the model's performance
print(classification_report(y_test, y_pred_classes))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

# Calculate ROC-AUC score
# Calculate test accuracy
train_accuracy = accuracy_score(y_train, y_pred_train_classes)
test_accuracy = accuracy_score(y_test, y_pred_classes)
print("Train Accuracy: ", np.round(train_accuracy, 4)*100, "%")
print("Test Accuracy: ", np.round(test_accuracy, 4)*100, "%")
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("ROC-AUC Score:", np.round(roc_auc, 4)*100, "%")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Create a DataFrame from predicted classes and IDs
df_results = pd.DataFrame({
    'ID': id_df_test2,
    'Target': df_test2_pred_classes.flatten()
})

# Save the DataFrame to CSV
df_results.to_csv('resultsOfAnaniDJATO.csv', index=False)

# Display the DataFrame
df_results.head(30)


# # Interpretation of Logistic Regression Model

# In[41]:


result_table = [
    ["Precision",	0.83,	0.76,	0.80],
    ["Recall",	0.76,	0.84,	0.80],
    ["F1-score",	0.79,	0.80,	0.80]
]

col = ["Metric",	"Class 0",	"Class 1",	"Overall Average"]

# Create a DataFrame from the
df_results = pd.DataFrame(result_table, columns=col)

# Display the DataFrame
df_results


# In[40]:


print("""
The results indicate that Dense Neural Network model has a good overall performance with a 
balanced precision and recall. The test and train accuracies are similar (79.35% and 79.41% respectively), 
indicating that the model is not overfitting and has good generalization capabilities. The high ROC-AUC score 
(84.55%) and the shape of the ROC curve suggest that the model effectively discriminates between the classes.
""")


# ### Details
# 
# **79% of all predictions were correct.**
# 
# #### Area Under the Receiver Operating Characteristic Curve (ROC AUC) :
# A score of **84.55%** indicates good ability to distinguish between the two classes.
# 
# 
# For **class 0**, the precision is **0.83**, meaning **83%** of the class **'no Internet access'** predictions are correct.   
# For **class 1**, the precision is **0.76**. So **76%** of class **'is Internet access'** predictions are correct.   
# 
# **Train accuracy = 79.41%**
# 
# **Test accuracy = 79.35%**
# So, the model fits the training data well without significant overfitting.
# Test accuracy is very close to the training accuracy, indicating good generalization ability of the model.
# 
# 
# The ROC-AUC score is **84.55%**, which is a good indication of the model’s 
# overall ability to distinguish between the positive and negative classes

# ## Dense Neural Network model

# In[36]:


# Define the model
model = Sequential()
model.add(Dense(128, input_dim=4041, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model and capture the training history
history = model.fit(X_train, y_train, epochs=300, batch_size=50, validation_split=0.2, verbose=1)

# Plot training and validation loss
plt.figure(figsize=(12, 5))
#plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Plot training and validation accuracy
plt.figure(figsize=(12, 5))
#plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.show()

# Predictions
y_pred_train_prob = model.predict(X_train).flatten()
y_pred_train_classes = (y_pred_train_prob >= 0.5).astype(int)

y_pred_prob = model.predict(X_test).flatten()
y_pred_classes = (y_pred_prob >= 0.5).astype(int)

df_test2_pred_prob = model.predict(df_test_val).flatten()
df_test2_pred_classes = (df_test2_pred_prob >= 0.5).astype(int)

# Model evaluation
print(classification_report(y_test, y_pred_classes))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

train_accuracy = accuracy_score(y_train, y_pred_train_classes)
test_accuracy = accuracy_score(y_test, y_pred_classes)
print("Train Accuracy: ", np.round(train_accuracy, 4) * 100, "%")
print("Test Accuracy: ", np.round(test_accuracy, 4) * 100, "%")

roc_auc = roc_auc_score(y_test, y_pred_prob)
print("ROC-AUC Score:", np.round(roc_auc, 4) * 100, "%")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Save the results
df_results = pd.DataFrame({
    'ID': id_df_test2,
    'Target': df_test2_pred_classes.flatten()
})
df_results.to_csv('resultsOfAnaniDJATO.csv', index=False)
df_results.head(30)


# # Interpretation of Dense Neural Network Model

# In[37]:


result_table = [
    ["Precision",	0.91,	0.75,	0.83],
    ["Recall",	0.70,	0.93,	0.82],
    ["F1-score",	0.79,	0.83,	0.81]
]

col = ["Metric",	"Class 0",	"Class 1",	"Overall Average"]

# Create a DataFrame from the
df_results = pd.DataFrame(result_table, columns=col)

# Display the DataFrame
df_results


# In[38]:


print("""
The results indicate that Dense Neural Network model has a good overall performance with a 
balanced precision and recall. The test and train accuracies are similar (81.30% and 80.91% respectively), 
indicating that the model is not overfitting and has good generalization capabilities. The high ROC-AUC score 
(83.03%) and the shape of the ROC curve suggest that the model effectively discriminates between the classes.
""")


# ### Details
# 
# **81% of all predictions were correct.**
# 
# #### Area Under the Receiver Operating Characteristic Curve (ROC AUC) :
# A score of **83.03%** indicates good ability to distinguish between the two classes.
# 
# 
# For **class 0**, the precision is **0.91**, meaning **91%** of the class **'no Internet access'** predictions are correct.   
# For **class 1**, the precision is **0.75**. So **75%** of class **'is Internet access'** predictions are correct.   
# 
# **Train accuracy = 80.91%**
# 
# **Test accuracy = 81.30%**
# So, the model fits the training data well without significant overfitting.
# Test accuracy is very close to the training accuracy, indicating good generalization ability of the model.
# 
# 
# The ROC-AUC score is **83.03%**, which is a good indication of the model’s 
# overall ability to distinguish between the positive and negative classes

# # Comparaison of Logistic Regression and Dense Neural Network models
# 
# En termes de taux de généralisation, le modèle de Réseau de neurones denses est legèrement mieux que le modèle de Regression Logistique.
# En effet, il y a un taux de précision de 81% pour le modèle de réseau de neurones dense et de 79% pour le modèle de regression logistique.
# 
# Cependant, si on regarde l'Aire sous la courbe ROC (ROC AUC), le modèle de regression logistique est légèrement plus performant que le modèle de réseaux de neurones dense : **84.55%** contre **83.03%**.
# 
# #### Pourquoi pour le moi le modèle de réseau de neurones est mieux ?
# - D'abord, il n'y a pas trop de différence entre les métriques
# - En plus, lorsque le volume des données sur lesquelles on a entrainé les 2 modèles va augementer, le modèle de réseau de neurones denses va pouvoir apprendre de lui même et améliorer tout seul ses performances.

# # Comparison of Logistic Regression and Dense Neural Network Models
# 
# In terms of generalization rate, the Dense Neural Network model is slightly better than the Logistic Regression model. 
# In fact, the Dense Neural Network model has an accuracy rate of **81%**, while the Logistic Regression model has an accuracy rate of **79%**.
# 
# However, if we look at the **Area Under the ROC Curve (ROC AUC)**, the Logistic Regression model slightly outperforms the Dense Neural Network model: **84.55%** versus **83.03%**.
# 
# ### Why do I think the neural network model is better?
#     - First, there is not a big difference between the metrics of the 2 models. 
#     
#     - In addition, when the volume of data on which the 2 models have been trained increases, 
#     the dense neural network model will be able to learn on its own and improve its performance on its own.
