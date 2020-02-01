from svm_source import *

# Classification data set
cancer = datasets.load_breast_cancer()

# Results for features & target names
#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

# Training & Testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# Results for training
classes = ['malignant','benign']

# C = the soft margin for prediction / double amount of points to calc
#clf = svm.SVC(kernel='linear', C=5)
clf = KNeighborsClassifier(n_neighbors=15)
# need better processing to use poly, will work on obtaining/creating a machine
clf.fit(x_train,y_train)

y_prediction = clf.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_prediction)
print(accuracy)