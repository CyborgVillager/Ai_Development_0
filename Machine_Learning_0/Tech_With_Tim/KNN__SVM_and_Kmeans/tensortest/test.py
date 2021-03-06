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
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)


'''
# Re-Training Model for the students
best = 0
for _ in range(35):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

    # Training Model
    linear = linear_model.LinearRegression()
    # linear.fit will fit the data -> x_train, y_train to find the 'best fit' line, it will then store the info
    # on linear = easier to test it
    linear.fit(x_train, y_train)

    # score the info / test the accuracy of it
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
        best = accuracy
    with open('studentmodels.pickle', 'wb') as file:
        pickle.dump(linear, file)
'''

# 1/30/20 file loads/open & start @ pickle_in = open('studentmodels.pickle', 'rb')
# the program skips the training process and heads straight to getting the data & saving it

# Now just loads in the model that was created aka studentmodels.pickle

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

# Basic design for the plots, this will be a scatter plot #
# Link for more info: https://matplotlib.org/gallery/style_sheets/ggplot.html
# As of now the points are overlapping with over students

# Plot Data to use for pyplot.scatter(data[],data[])
plot0 = 'G1'
plot1 = 'G3'
studytime_plot2 = 'studytime'
failures_plot3 = 'failures'
absences_plot4 = 'absences'
style.use("ggplot")
# Input data from Plot Data
pyplot.scatter(data[absences_plot4],data[plot1])
# Labels for the graph x & y
pyplot.xlabel(absences_plot4)
pyplot.ylabel('Final Grades')
# Results
pyplot.show()