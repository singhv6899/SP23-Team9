
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import numpy as np
from sklearn.inspection import permutation_importance
import os
import nltk


nltk.download('stopwords')
nltk.download('punkt')


class WebexFeatureClassifier:

    def load(self):
        #please make sure to place all the files in a webex folder
        # Set the directory path containing the doc files
        old_months= ['March', 'Feb', 'Jan']
        
        df=pd.read_csv('data/webex.csv')

        """## Data Preprocessing """

        #droing the null values 
        df=df.dropna()
        # removing the stop values 
        stopwords_list = stopwords.words('english')
        tokenizer = CountVectorizer().build_tokenizer()

        def tokenize(text):
            tokens = word_tokenize(text)
            return [token for token in tokens if token not in stopwords_list and token.isalpha()]

        # Create a document-term matrix using count vectorizer
        self.vectorizer = CountVectorizer(tokenizer=tokenize)

        X = self.vectorizer.fit_transform(df['Description'])
        y = df['Label']
        self.feature_names = self.vectorizer.get_feature_names_out()

        return X,y

        # Train test split of the feature dataset

    def model(self,kernel,X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        """### SVM with Sigmoid kernal"""

        # create and fit SVM with sigmoid kernal
        clf = SVC(kernel=kernel)

        # Fit the classifier to the training data
        clf.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = clf.predict(X_test)

        # evaluate performance on testing set

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # printing 
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        # Compute permutation feature importance
        perm_importance = permutation_importance(clf, X_train.toarray(), y_train, n_repeats=2, random_state=42)

        # Get the indices of the top 10 features with the highest importances (for predicting "new" features)
        new_feature_indices_rbf = perm_importance.importances_mean.argsort()[::-1][:10]
        new_feature_words_rbf = [self.feature_names[i] for i in new_feature_indices_rbf]
        print("Top 10 words for predicting 'new' features using SVM sigmoid kernel:")
        print(new_feature_words_rbf)

        # Get the indices of the top 10 features with the lowest importances (for predicting "old" features)
        old_feature_indices_rbf = perm_importance.importances_mean.argsort()[:10]
        old_feature_words_rbf = [self.feature_names[i] for i in old_feature_indices_rbf]
        print("Top 10 words for predicting 'old' features using SVM sigmoid kernel:")
        print(old_feature_words_rbf)

        return clf, new_feature_words_rbf, old_feature_words_rbf
    def predict(self,clf, new_string):
        # Convert the new string to a list with a single element
        new_string_list = [new_string]

        # Vectorize the new string
        new_string_vectorized = self.vectorizer.transform(new_string_list)

        # Make a prediction using the trained classifier
        prediction = clf.predict(new_string_vectorized)

        # Return the prediction (as a string)
        if prediction[0] == 0:
            return "Old feature"
        else:
            return "New feature"
