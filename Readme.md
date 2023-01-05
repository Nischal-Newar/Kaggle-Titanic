## Kaggle - Titanic

In this challenge, I trained different classification models. 
In the end I selected the best model and made the prediction.
###### Best Model: Random Forest Classifier
###### Accuracy: 76%

### Steps:
- Understand the data-set by visualizing or using tabular representation
- After understanding the data-set, pre-process the data
- Drop the unnecessary features and add the missing values for columns by taking the median.
- Encoded the categorical feature Sex and Embarked to integers
- Divided the data-set into 80% training and 20% test
- Trained 7 different classification models using the training set
- Tested the 7 different classification models using the test set
- Selected Random Forest Classifier on the basis of above observation
- Tested the model using the provided data-set

### Some Important Notes:
* I could further improve the accuracy by combining features such as SibSp and Parch.
* I might use important features when using Random Forest Classifier
* Another option would be to find the best values for max_depth, min_sample_split, criterion etc
