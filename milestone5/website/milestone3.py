
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

import nltk

nltk.download('stopwords')
nltk.download('punkt')


class ZoomFeatureClassifier:

    def load(self):
        #fetching the sheetnames from the excel file
        tabs = pd.ExcelFile("data/Zoom-features-2022.xlsx").sheet_names 
        tabs

        #initilizing the column values
        self.feature_data=[]
        self.feature_names=[]
        labels=[]
        old_months= [ 'June-2022','May-2022', 'April-2022', 'March-2022', 'Feb-2022','Jan-2022']

        #reading the data from the excel files 
        for i in tabs:
            data= pd.read_excel('data/Zoom-features-2022.xlsx',sheet_name=i)
            if i in old_months:
                labels.extend([0]*len(data['Feature Title'].values))
            else:
                labels.extend([1]*len(data['Feature Title'].values))
            self.feature_data.extend((data['Feature Description'].values))
            self.feature_names.extend((data['Feature Title'].values))

        # creating a data frame to store the column values 
        df=pd.DataFrame({"Title":self.feature_names,"Description": self.feature_data,"Label":labels})
        df["Description"]=df["Description"].apply(str)
        df["Description"]

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
        new_string_list = ["new  feature"]

        # Vectorize the new string
        new_string_vectorized = self.vectorizer.transform(new_string_list)

        # Make a prediction using the trained classifier
        prediction = clf.predict(new_string_vectorized)

        # Return the prediction (as a string)
        if prediction[0] == 0:
            print("Old feature")
        else:
            print("New feature")

        # Return the prediction (as a string)
        if prediction[0] == 0:
            return "Old feature"
        else:
            return "New feature"
