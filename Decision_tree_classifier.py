"""
Kara Shibley & Griffin Reichmuth
CSE 163 AC
This file builds a ML classifier to classify county type.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot


data_c = pd.read_csv("features_labels")
data_c = data_c.drop(["Unnamed: 0"], axis=1)

features = data_c.loc[:, data_c.columns != "2013 code"]
labels = data_c["2013 code"]


features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3)

model = DecisionTreeClassifier(max_depth=5)


model.fit(features_train, labels_train)

# Compute training accuracy
train_predictions = model.predict(features_train)
print('Train Accuracy:', accuracy_score(labels_train, train_predictions))

# Compute test accuracy
test_predictions = model.predict(features_test)
print('Test  Accuracy:', accuracy_score(labels_test, test_predictions))


# get importance
importance = model.feature_importances_
# summarize feature importance
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
