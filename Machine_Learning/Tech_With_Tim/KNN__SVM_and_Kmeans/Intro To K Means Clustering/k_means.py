from kmeans_source import *
# link for more info -> https://techwithtim.net/tutorials/machine-learning-python/k-means-2/
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# https://scikit-learn.org/stable/modules/clustering.html
# https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
digits = load_digits()
# .data = is the features of the program
# digits by the default its a big too big for the computer, so the program will
# scale it to ease of the processing of it
data = scale(digits.data)
y = digits.target

# number of centrios for the system to make
# 15 digits
k = 15
# the amount of 'features' or instances = to the data.shape
samples, features = data.shape

# Scikit Learn Example's Imported
# the y @ line 8 value will compare with the labels from estimator.labels
# the program will auto-make the data/predict the data
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))
# euclidean distance

# classifier/ prints out the result of the info
# init - number of times the algorith will run
clf = KMeans(n_clusters=k, init="random", n_init=10)
print('Results from line 23-29: ')
bench_k_means(clf, "1", data)
