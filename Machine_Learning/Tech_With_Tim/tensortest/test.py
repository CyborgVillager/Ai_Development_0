from tim_lesson_source import *

data = pandas.read_csv('../Excel/student-mat.csv', sep=';')
# Known as an atribute
data = data[['G1','G2','G3','studytime','sex','age','health','Pstatus','failures','absences']]

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