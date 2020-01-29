from tim_lesson_source import *

data = pandas.read_csv('../Excel/student-mat.csv', sep=';')
data = data[['G1','G2','G3','studytime','failures','absences']]
print(data)
# print 1st 5 elements -> data.head()
print('---------    ---------   ---------   ---------')
print(data.head())
