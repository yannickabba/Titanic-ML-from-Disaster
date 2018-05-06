# -*- coding: utf-8 -*-
"""
Kaggle: titanic
@author: Yannick Abba

predict if a passenger survived the sinking of the Titanic or not. 

PassengerId -- A numerical id assigned to each passenger.
Survived -- Whether the passenger survived (1), or didn't (0). We'll be making predictions for this column.
Pclass -- The class the passenger was in -- first class (1), second class (2), or third class (3).
Name -- the name of the passenger.
Sex -- The gender of the passenger -- male or female.
Age -- The age of the passenger. Fractional.
SibSp -- The number of siblings and spouses the passenger had on board.
Parch -- The number of parents and children the passenger had on board.
Ticket -- The ticket number of the passenger.
Fare -- How much the passenger paid for the ticker.
Cabin -- Which cabin the passenger was in.
Embarked -- Where the passenger boarded the Titanic 


Results          Model  Score  Kaggle
0        Random Forest  84.36  0.75598
1  Logistic Regression  83.80
3            XGB Trees  82.68
2           Kernel SVM  63.69

"""

# import libraries
import pandas as pd
pd.set_option('display.max_columns', None)
import re
import seaborn as sns

# define working directory
PATH = r"C:\Users\s7102976\Documents\1_Data Science\4_Templates\1_python\titanic"

# read data
df_train = pd.read_csv(f'{PATH}\\train.csv', index_col='PassengerId')

# explore data
df_train.info()

# modify data types
df_train['Survived'] = df_train['Survived'].astype(str)
df_train['Pclass'] = df_train['Pclass'].astype(str)

# define target variable
target = df_train['Survived']
df_train = df_train.drop('Survived',axis=1)

# explore distributions
df_train.describe()

# check for missing data
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df_train.isnull().mean()
df_train.isnull().sum()

# impute missing data - Age
df_train.groupby(['Sex','Pclass']).median()['Age']

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    Sex = cols[2]
    
    if pd.isnull(Age):
        if Sex == 'female':
            if Pclass == 1:
                return 35
            elif Pclass == 2:
                return 28
            else:
                return 22       
        else:
            if Pclass == 1:
                return 40
            elif Pclass ==2:
                return 30
            else:
                return 25       
    else:
        return Age

df_train.Age = df_train[['Age','Pclass','Sex']].apply(impute_age,axis=1)

# impute missing data - Embarked
df_filter = df_train[df_train.Embarked.isnull()] # Embarked only has 2 missing values, both missing values are 1st class tickets at $80
df_train.groupby(['Embarked','Pclass']).median()['Fare'] # Likely, both passengers Embarked at 'C'  
df_train.Embarked = df_train.Embarked.fillna(value='C')

# impute missing data - Cabin
def simplify_cabin(df):
    df.Cabin = df.Cabin.fillna('Missing')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

df_train = simplify_cabin(df_train)
df_train.Cabin.value_counts()

# feature engineering - title
def extract_title(df):
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.')
    return df 

df_train = extract_title(df_train)
df_train.Title.value_counts()

df_train.Title = df_train.Title.replace(['Dr','Rev','Col','Major','Don','Lady','Countess','Jonkheer','Capt','Sir'],'Rare')
pd.crosstab(df_train.Sex,df_train.Title)

df_train.Title = df_train.Title.replace('Mlle','Miss')
df_train.Title = df_train.Title.replace('Ms','Miss')
df_train.Title = df_train.Title.replace('Mme','Mrs')

# feature engineering - name
df_train['Surname'] = df_train['Name'].apply(lambda x: x.split(',')[0])

# feature engineering - family
df_train['Familysize'] = df_train['SibSp'] + df_train['Parch'] + 1
df_train['Family'] = df_train.Surname +'_'+ df_train['Familysize'].astype(str)

# feature engineering - age
def simplify_age(df):
    bins = (-1, 0, 14, 24, 65, 120)
    group_names = ['Unknown', 'Child', 'Youth','Adult','Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df
df_train = simplify_age(df_train)

# feature engineering - is a parent
df_train['Parent'] = 'No'
        
def impute_parent(cols):
    Age = cols[0]
    Parch = cols[1]
    Parent = cols[2]
    
    if Parent == 'No':
        if Age in ('Adult','Senior'):
            if Parch > 0:
                return 'Yes' 
            else:
                return Parent
        else:
            return Parent
    else: 
        return Parent

df_train.Parent = df_train[['Age','Parch','Parent']].apply(impute_parent,axis=1)        

# feature engineering - is married
df_train['Married'] = 'No'

def impute_married(cols):
    Age = cols[0]
    SibSp = cols[1]
    Married = cols[2]
    
    if Married == 'No':
        if Age in ('Adult','Senior'):
            if SibSp > 0 :
                return 'Yes' 
            else:
                return Married
        else:
            return Married
    else: 
        return Married

df_train.Married = df_train[['Age','SibSp','Married']].apply(impute_married,axis=1)  
 

# feature engineering - ticket
df_train['TicketD'] = 'num'

def impute_ticket(cols):
    Ticket = cols[0]
    TicketD = cols[1]
    
    if TicketD == 'num':
        if bool(re.search('[a-zA-Z]', Ticket)):
            return 'char'
        else:
            return TicketD
    else: 
        return TicketD
        
df_train.TicketD = df_train[['Ticket','TicketD']].apply(impute_ticket,axis=1)    

# feature engineering - fare
def simplify_fare(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

df_train = simplify_fare(df_train)

# remove redundant columns
df_train = df_train.drop(['Name','SibSp','Parch','Ticket','Surname','Familysize'],axis=1)

# encode variables
df_train = pd.get_dummies(df_train, drop_first=True)

# split data
X = df_train
y = target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


""" train models """
# train random forest
from sklearn.ensemble import RandomForestClassifier
trees = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
trees.fit(X_train, y_train)

# fitting xgboost trees
from xgboost import XGBClassifier
xgb = XGBClassifier(objective='binary:logistic', n_estimators=100)
xgb.fit(X_train,y_train)

# train logistic regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# train Kernel SVM
from sklearn.svm import SVC
kernel = SVC(kernel = 'rbf', random_state = 0)
kernel.fit(X_train, y_train)

""" evaluate models """
# compute accuracy scores, (60% is acceptable, 70% is good, 85% is very good)
acc_rfc = round(trees.score( X_test , y_test ) * 100, 2)
acc_xgb = round(xgb.score(X_test, y_test) * 100, 2)
acc_log = round(logmodel.score( X_test, y_test ) * 100, 2)
acc_svm = round(kernel.score(X_test, y_test) * 100, 2)

models = pd.DataFrame({
        'Model':['Random Forest','XGB Trees','Logistic Regression','Kernel SVM'],
        'Score':[acc_rfc, acc_xgb, acc_log, acc_svm] })

models.sort_values(by='Score', ascending=False)


""" test models """
pred_rfc = trees.predict(X_test) # random forest
pred_xgb = xgb.predict(X_test) # xgb trees
pred_log = logmodel.predict(X_test) # logistic regression
pred_svm = kernel.predict(X_test) # kernel svm

""" assess models """
from sklearn.metrics import classification_report

print(classification_report(y_test,pred_rfc)) # random forest
print(classification_report(y_test,pred_xgb)) # xgb trees
print(classification_report(y_test, pred_log)) # logistic regression
print(classification_report(y_test,pred_svm)) # kernel SVM



""" test data pipeline """

df_test = pd.read_csv(f'{PATH}\\test.csv', index_col='PassengerId')
df_test['Pclass'] = df_test['Pclass'].astype(str)
df_test.Age = df_test[['Age','Pclass','Sex']].apply(impute_age,axis=1)
df_test = simplify_cabin(df_test)
df_test = extract_title(df_test)
df_test.Title.value_counts()
df_test.Title = df_test.Title.replace(['Dr','Rev','Col','Major','Don',
                                         'Lady','Countess','Jonkheer','Capt','Sir',
                                         'Dona'],'Rare')
df_test.Title = df_test.Title.replace('Mlle','Miss')
df_test.Title = df_test.Title.replace('Ms','Miss')
df_test.Title = df_test.Title.replace('Mme','Mrs')
df_test['Surname'] = df_test['Name'].apply(lambda x: x.split(',')[0])
df_test['Familysize'] = df_test['SibSp'] + df_test['Parch'] + 1
df_test['Family'] = df_test.Surname +'_'+ df_test['Familysize'].astype(str)
df_test = simplify_age(df_test)
df_test['Parent'] = 'No'
df_test.Mother = df_test[['Age','Parch','Parent']].apply(impute_parent,axis=1)  
df_test['Married'] = 'No'
df_test.Married = df_test[['Age','SibSp','Married']].apply(impute_married,axis=1) 
df_test['TicketD'] = 'num'
df_test.TicketD = df_test[['Ticket','TicketD']].apply(impute_ticket,axis=1)
df_test = simplify_fare(df_test)
df_test = df_test.drop(['Name','SibSp','Parch','Familysize','Ticket','Surname'],axis=1)
df_test = pd.get_dummies(df_test, drop_first=True)

missing_cols = set(df_train.columns)-set(df_test.columns)
for c in missing_cols:
    df_test[c]='0'
df_test = df_test[df_train.columns]
df_test = df_test.astype(str)

""" predict the actual Test data """
# run best model (random forest)
predictions = trees.predict(df_test)
output = pd.DataFrame({ 'PassengerId' : df_test.index.values, 'Survived': predictions })
output.head()
output.to_csv('titanic_predictions_v6.csv', index = False)
