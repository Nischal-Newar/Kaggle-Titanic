# import libraries
import matplotlib.pyplot as plot
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

"""
    First:  analyze the data-set (check mean, max values or visualize the correlation etc)
    Second: process the data set such as drop the features that can be ignored, add/remove the Null values 
            (missing data) of a feature, divide the training set into test and training set
    Third:  decide which classification algorithm to use: Logistic Regression, Random Forest etc
    Fourth: train the model using training set
    Fifth:  test the model using the test set
"""


# Classification Machine Learning Models
def models(x_train, y_train):
    # Using Logistic Regression Algorithm to the Training Set
    log = LogisticRegression(random_state=0)
    log.fit(x_train, y_train)

    # Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn.fit(x_train, y_train)

    # Using SVC method of svm class to use Support Vector Machine Algorithm
    svc_lin = SVC(kernel='linear', random_state=0)
    svc_lin.fit(x_train, y_train)

    # Using SVC method of svm class to use Kernel SVM Algorithm
    svc_rbf = SVC(kernel='rbf', random_state=0)
    svc_rbf.fit(x_train, y_train)

    # Using GaussianNB method of naive_bayes class to use Naive Bayes Algorithm
    gauss = GaussianNB()
    gauss.fit(x_train, y_train)

    # Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(x_train, y_train)

    # Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
    forest = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=8, min_samples_split=5,
                                    random_state=0)
    forest.fit(x_train, y_train)

    # print model accuracy on the training data.
    print('Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    print('K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
    print('Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
    print('Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
    print('Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
    print('Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    print('Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))

    return log, knn, svc_lin, svc_rbf, gauss, tree, forest


class titanicKaggle:

    # initial function to read the data set
    def __init__(self, path):
        self.data_set = pd.read_csv(path)

    # function to show correlation between features
    def show_graph(self):
        cols = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked', 'Survived']
        n_rows = 2
        n_columns = 3
        fig, axis = plot.subplots(n_rows, n_columns, figsize=(n_columns * 3.2, n_rows * 3.2))
        for i in range(0, n_rows):
            for j in range(0, n_columns):
                ii = i * n_columns + j  # iterate through each column of cols
                ax = axis[i][j]  # show where to position each sub-plot
                sns.countplot(x=self.data_set[cols[ii]], hue=self.data_set['Survived'], ax=ax)
                ax.set_title(cols[ii])
                ax.legend(title='Survived', loc='best')
        plot.tight_layout()
        plot.show()
        return None

    def drop_features(self, features):
        self.data_set = self.data_set.drop(labels=features, axis=1)
        return None

    def fill_na_values(self, features):
        for feature in features:
            self.data_set[feature] = titanic.data_set[feature].fillna(titanic.data_set[feature].median())
        return None

    def label_encoder(self, feature):
        labelencoder = LabelEncoder()
        self.data_set[feature] = labelencoder.fit_transform(self.data_set[feature].values)


# main function to execute the series of operations
if __name__ == "__main__":
    # import the training file
    titanic = titanicKaggle('./titanic/train.csv')

    # check the datatypes
    # print(titanic.data_set.dtypes)

    # graph to see relation between features and survived column
    # titanic.show_graph()

    """
        In the above graph, the number of males died were higher in compare to female. So, I am using the table 
        representation to analyze the survival rate of different sex with respect to Pclass, Embarked and Age.
    """
    # print('---------------------------------------------------------------------------------')
    # print(titanic.data_set.pivot_table(values='Survived', index='Sex', columns='Pclass'))
    # print('---------------------------------------------------------------------------------')
    # print(titanic.data_set.pivot_table(values='Survived', index='Sex', columns='Embarked'))
    # print('---------------------------------------------------------------------------------')
    # Age = pd.cut(titanic.data_set['Age'], [0, 18, 100])
    # print(titanic.data_set.pivot_table(values='Survived', index='Sex', columns=Age))
    # print('---------------------------------------------------------------------------------')

    # check the number of na values for each feature's
    # print(titanic.data_set.isna().sum())

    """ 
        drop the Cabin feature since most of the values are na
        drop the Ticket feature (no direct relation to rate of survival)
    """
    titanic.drop_features(['Cabin', 'Ticket', 'Name'])

    """
        filling the na values of Age and Embarked with median value
    """
    titanic.fill_na_values(['Age'])

    """
        Encoding categorical data values (Transforming object data types to integers)
        Encode sex column
        Encode embarked
    """
    titanic.label_encoder('Sex')
    titanic.label_encoder('Embarked')

    # Split the data into independent 'X' and dependent 'Y' variables
    Y = titanic.data_set['Survived'].values
    X = titanic.data_set.drop('Survived', axis=1)

    # Split the dataset into 80% Training set and 20% Testing set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = models(x_train=X_train, y_train=Y_train)

    print('---------------------------------------------------------------------------------------')
    model_name = ['Logistic Regression', 'K Nearest Neighbor', 'Support Vector Machine Linear',
                  'Support Vector Machine RBF', 'Gaussian Naive Bayes', 'Decision Tree', 'Random Forest Classifier']
    for i in range(len(model)):
        print(model_name[i], 'Testing Accuracy:', model[i].score(X_test, Y_test))

    # Testing the actual test data
    actual_test_data = titanicKaggle('./titanic/test.csv')

    # cleaning the records
    actual_test_data.drop_features(['Cabin', 'Ticket', 'Name'])
    actual_test_data.fill_na_values(['Age', 'Fare'])
    actual_test_data.label_encoder('Sex')
    actual_test_data.label_encoder('Embarked')
    X_final_test = sc.fit_transform(actual_test_data.data_set)

    # selecting random forest because it gave the best prediction with the training data-set
    predictions = model[6].predict(X_final_test)
    final_df = pd.DataFrame(actual_test_data.data_set['PassengerId'])
    final_df['Survived'] = predictions
    final_df.to_csv('./titanic/predictions.csv', index=False)
