from knn_source import *

# statements for data[]
buying = 'buying'
maint = 'maint'
persons = 'persons'
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
lug_boot = labencode.fit_transform(list(data[lug_boot]))
safety = labencode.fit_transform(list(data[safety]))
cls = labencode.fit_transform(list(data[cls]))

# Prediction Varia
prediction = 'class'

# Result
print(buying)

