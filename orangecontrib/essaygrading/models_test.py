from orangecontrib.essaygrading.widgets.OWScore import quadratic_weighted_kappa
from orangecontrib.essaygrading.widgets.OWAttributeSelection import calculateAttributes
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

from Orange.data import Table, Domain, ContinuousVariable
from Orange.regression import RandomForestRegressionLearner
from Orange.regression import LinearRegressionLearner

# table = Table.from_file("datasets/FAS/set1_train_2.tab")
#table = Table.from_file("datasets/FAS/set1_train_2.tab")
table = Table.from_file("datasets/FAS/set8_AGE_tfidf.tab")

scores = np.array(table.Y)
ALL_ATTRIBUTES = np.array([x for x in table.X])
ALL_ATTRIBUTES = np.nan_to_num(ALL_ATTRIBUTES)
print(table)

print(ALL_ATTRIBUTES.shape)

scaler = preprocessing.MinMaxScaler()
#ALL_ATTRIBUTES = scaler.fit_transform(ALL_ATTRIBUTES)

# ALL_ATTRIBUTES = preprocessing.normalize(ALL_ATTRIBUTES.transpose()).transpose()

# Shuffle array
indices = np.arange(len(ALL_ATTRIBUTES))
np.random.shuffle(indices)
ALL_ATTRIBUTES = ALL_ATTRIBUTES[indices]
scores = scores[indices]


print(ALL_ATTRIBUTES)
print(max(ALL_ATTRIBUTES[:, 5]))
print(min(ALL_ATTRIBUTES[:, 5]))

# ALL_ATTRIBUTES = ALL_ATTRIBUTES[:, :58]

k = 10
chunk = int(len(ALL_ATTRIBUTES) / k)

attributes = np.array([ALL_ATTRIBUTES[:, 0]]).transpose()
kept_attributes = []
avg_kappa = 0

train = []
train_scores = []
test = []
test_scores = []

for j in range(10):
    start = j * chunk
    end = min((j + 1) * chunk, len(ALL_ATTRIBUTES))
    train.append(np.concatenate((ALL_ATTRIBUTES[:start], ALL_ATTRIBUTES[end:])))
    train_scores.append(np.concatenate((scores[:start], scores[end:])))
    test.append(ALL_ATTRIBUTES[start:end])
    test_scores.append(scores[start:end])




added_attribute = None
best_score = -1
best_i = -1
curr_kappa = 0

kappas = []

for j in range(k):

    train_attributes = train[j]
    test_attributes = test[j]

    rf = LinearRegressionLearner()  # (n_estimators=100, )#, min_samples_split=max(int(len(attributes[0])/3), 2))
    #rf = RandomForestRegressionLearner(n_estimators=100, bootstrap=True, max_features=0.33)#, min_samples_split=max(int(len(attributes[0])/3), 2)))

    # print("Training...")
    rf = rf.fit(train_attributes, train_scores[j])

    # print("Predicting...")
    predictions = rf.predict(test_attributes)
    #print(predictions)

    #print(len(predictions))
    #print(len(test_scores[j]))

    predictions = [p if p < max(test_scores[j]) else max(test_scores[j]) for p in predictions]

    # print("Results...")
    kappas.append(quadratic_weighted_kappa(np.round(predictions), test_scores[j]))

print("FINISHED")
print(kappas)
curr_kappa = sum(kappas) / k
print(str(curr_kappa))

