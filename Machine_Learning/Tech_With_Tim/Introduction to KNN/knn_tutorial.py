from knn_source import *

# statements for data[]
buying = 'buying'
maint = 'maint'
persons = 'persons'
door = 'door'
lug_boot = 'lug_boot'
safety = 'safety'
cls = 'class'

data = pd.read_csv('car_data_set/car.data')
print(data.head())



# new objects these will return a numpy array
# turn
labencode = preprocessing.LabelEncoder()
# making an array for each column
buying = labencode.fit_transform(list(data[buying]))
maint = labencode.fit_transform(list(data[maint]))
persons = labencode.fit_transform(list(data[persons]))
door = labencode.fit_transform(list(data[door]))
lug_boot = labencode.fit_transform(list(data[lug_boot]))
safety = labencode.fit_transform(list(data[safety]))
cls = labencode.fit_transform(list(data[cls]))



# Result
#print(buying)

# Prediction Varia
prediction = 'class'


X = list(zip(buying,maint,door, persons, lug_boot, safety))
Y = list(cls)

# Splitting the test, can be found @ tensortest/test.py
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
print(x_train,y_test)