from tim_lesson_source import *

data = pandas.read_csv('../Excel/student-mat.csv', sep=';')
# Known as an atribute
data = data[['G1','G2','G3','studytime','failures','absences']]

# Result
print('---------    ---------   ---------   ---------')
#print(data.head())

# G3 = Final Grade
# program will 'predict' the atribute
predictiction = 'G3'

# The Features
X = numpy.array(data.drop([predictiction],1))
# Labels
Y = numpy.array(data[predictiction])

# The program will now train it self 4 90% & do test 4 10%
# Line 22 takes all of the attributes from Features & Labels -> the program will split them up into 4 arrays
# x_train -> section of numpy.array(data.drop([predictiction],1)) , y_train -> numpy.array(data[predictiction])
# x_test & y_test will test the accuracy of the program/model.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
# Training Model
linear = linear_model.LinearRegression()
# linear.fit will fit the data -> x_train, y_train to find the 'best fit' line, it will then store the info
# on linear = easier to test it
linear.fit(x_train, y_train)

# score the info / test the accuracy of it
accuracy = linear.score(x_test, y_test)
print(accuracy)

# open / create a file called studentmodels and save it by using pickle.dump in the directory
# https://docs.python.org/3/library/pickle.html
# Side note for future reference pickle is not secure, so for future files make sure its clean
with open('studentmodels.pickle', 'wb') as file:
    pickle.dump(linear, file)
# Read the pickle file
pickle_in = open('studentmodels.pickle', 'rb')
# Loading the picking onto the linear model
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('The Bigger the Coefficient the bigger the attribute has for its value ')
print('------   ------  ------  ------  ------  ------  ------')

print('Intercept: \n', linear.intercept_)

# Now predicition will predict the student's grade
predictions = linear.predict(x_test)

# will test the array from the models that were not trained on and see
print('1st / Begin grade, Semester Grade, Hours of Study Time, Failures, Absences, Actual Grade')
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
