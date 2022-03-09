import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from IPython.display import display
from sklearn.metrics import confusion_matrix

def prediction(str):
    dataset = pd.read_csv('consumer_complaints.csv', dtype='unicode')
    col = ['product', 'issue']
    dataset = dataset[col]
    dataset = dataset[pd.notnull(dataset['issue'])]
    dataset.columns = ['product', 'issue']
    dataset['category_id'] = dataset['product'].factorize()[0]
    category_id_dataset = dataset[['product', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_dataset.values)
    id_to_category = dict(category_id_dataset[['category_id', 'product']].values)
    # dataset.groupby('product').issue.count().plot.bar(ylim=0)

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(dataset.issue).toarray()
    labels = dataset.category_id

    N = 2
    for product, category_id in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

    X_train, X_test, y_train, y_test = train_test_split(dataset['issue'], dataset['product'], random_state=0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    model = LinearSVC()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, dataset.index, test_size=0.40, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)

    for predicted in category_id_dataset.category_id:
        for actual in category_id_dataset.category_id:
            if predicted != actual and conf_mat[actual, predicted] >= 10:
                print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
                display(dataset.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['product', 'issue']])
                print('')
    model.fit(features, labels)
    N = 2
    for product, category_id in sorted(category_to_id.items()):
        indices = np.argsort(model.coef_[category_id])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
        bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]

    return clf.predict(count_vect.transform([str]))
    return (metrics.classification_report(y_test, y_pred, target_names=dataset['product'].unique()))