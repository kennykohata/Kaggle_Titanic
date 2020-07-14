# Description: This program predicts if a passenger will survive on the Titanic

# Data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

# Load the datasets 
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(train_df.columns.values)

#Print the first 10 rows 
train_df.head(10)

# Count the number of rows and columns in the dataset
train_df.shape

# Get some statistics
train_df.describe()

# Get a count of the # of survivors
train_df['Survived'].value_counts()

# Visualize the # of survivors
sns.countplot(train_df['Survived']).set_title('# of Survivors')

# Visualize the count of survivors for columns 'sex', 'age', 'pclass', 'sibsp', 'parch', 'embarked'
cols = ['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Embarked']
n_rows = 2
n_cols = 3

# Establish Subplot Grid and Fig Size for each graph
fig, axs = plt.subplots(n_rows, n_cols, figsize = (n_cols * 5, n_rows * 5))

for r in range(0, n_rows):
    for c in range(0, n_cols):
        i = r*n_cols + c # index to go through the number of columns
        ax = axs[r][c] # show where to position each subplot
        sns.countplot(train_df[cols[i]], hue=train_df['Survived'], ax=ax)
        ax.set_title(cols[i])
        ax.legend(title='Survived', loc = 'upper right')

# Look at the survival rate by sex
train_df.groupby('Sex')[['Survived']].mean()

# Look at the survival rate by sex and class
train_df.pivot_table('Survived', index = 'Sex', columns = 'Pclass')

# Look at the survival rate by sex and class visually 
train_df.pivot_table('Survived', index = 'Sex', columns = 'Pclass').plot()

# Plot the survival rate of each class
sns.barplot(x = 'Pclass', y = 'Survived', data = train_df)

# Look at the survival rate by sex, age, and class (Break up age by kids and adults)
Age = pd.cut(train_df['Age'], [0,18,80]) 
train_df.pivot_table('Survived', ['Sex', Age], 'Pclass')

# Plot the prices paid of each class
plt.scatter(train_df['Fare'], train_df['Pclass'], color = 'red', label = 'Passenger Paid')
plt.ylabel('Class')
plt.xlabel('Fare Price')
plt.title('Price of Each Class')
plt.legend()
plt.show()

# Count the empty values in the columns
train_df.isnull().sum()

# Plot a heatmap of null values 
sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# Look at the values in each column and get a count 
for val in train_df:
    print(train_df[val].value_counts())
    print

# Drop the columns
train_df = train_df.drop(['PassengerId','Ticket', 'Name', 'Cabin'], axis=1)

#Remove the rows with missing values
train_df = train_df.dropna(subset =['Age', 'Embarked'])

#Count the NEW number of rows and columns in the data set
train_df.shape

# Look at the data types 
train_df.dtypes

#Print the unique values in the columns
print(train_df['Sex'].unique())
print(train_df['Embarked'].unique())

#Encoding categorical data values (Transforming object data types to integers)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

#Encode sex column
train_df.iloc[:,2]= labelencoder.fit_transform(train_df.iloc[:,2].values)
#print(labelencoder.fit_transform(titanic.iloc[:,3].values))

#Encode embarked
train_df.iloc[:,7]= labelencoder.fit_transform(train_df.iloc[:,7].values)
#print(labelencoder.fit_transform(titanic.iloc[:,8].values))

#Print the NEW unique values in the columns
print(train_df['Sex'].unique())
print(train_df['Embarked'].unique())

#Check the data type of the encoded Embarked column
train_df.dtypes

# Split the data into independent 'X' and dependent 'Y' variables
X = train_df.iloc[:, 1:8].values 
Y = train_df.iloc[:, 0].values

# Split the dataset in 80% training and 20% testing 
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Scale the data 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Create a funcation with different machine learning models
def models(X_train, Y_train):
    
    # Use Logistic Regression (binary regression)
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, Y_train)
    
    # Use KNeighbors (SVM used for classification based on k neighbors)
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
    knn.fit(X_train, Y_train)
    
    # Use SVC (Linear Kernerl) (SVM supervised model for classification)
    svc_lin = SVC(kernel = 'linear', random_state = 0)
    svc_lin.fit(X_train, Y_train)
    
    # Use SVC (RBF Kernerl) (SVM supervised model for classification)
    svc_rbf = SVC(kernel = 'rbf', random_state = 0)
    svc_rbf.fit(X_train, Y_train)
    
    # Use Gaussian NB (Naive Bayes predicts the probability of different class based on various attr)
    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)
    
    # Use Decision Tree (SVM to classify, continuously split to a certain parameter)
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, Y_train)
    
    # Use RandomForestClassifier (SVM building an ensemble of decision tress trained with bagging)
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, Y_train)
    
    #print model accuracy on the training data.
    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
    print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
    print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
    print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
    print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
    
    return log, knn, svc_lin, svc_rbf, gauss, tree, forest

# Get and train all of the models
model = models(X_train, Y_train)

# Show the confusion matrix and accuracy for all the models on the test data
from sklearn.metrics import confusion_matrix
for i in range(len(model)):
    
    cm = confusion_matrix(Y_test, model[i].predict(X_test))
    
    # Extract TN, FP, FN, TP
    TN, FP, FN, TP = cm.ravel()
    
    test_score = (TP + TN) / (TP + FP + FN + TN)
    
    print(cm)
    print('Model[{}] Testing Accuracy = "{}"'.format(i, test_score))
    print()

# Find the importances of each feature
forest = model[6]
importances = pd.DataFrame({'feature':train_df.iloc[:, 1:8].columns,'importance':np.round(forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances

#Visualize the importance
importances.plot.bar()

# Print the prediction of the random forest classifier 
pred = model[6].predict(X_test)
print(pred)
print()

#Print the actual values 
print(Y_test)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": pred
    })
# submission.to_csv('../output/submission.csv', index=False)


# Testing my survival rate 
#Survived      int64
#Pclass        int64
#Sex           int64
#Age         float64
#SibSp         int64
#Parch         int64
#Fare        float64
#Embarked      int64

# My survival attrs
my_survival = [[3,1,23,1, 0, 200, 0]]

# Scale my survival
sc = StandardScaler()
my_survival_scaled = sc.fit_transform(my_survival)

# Print prediciton of my surivival using Random Forest Classifier
pred = model[6].predict(my_survival_scaled)
print(pred)

if pred == 0:
    print('Unfortunately, you did not survive!')
else: 
    print('Congrats, you survived!')