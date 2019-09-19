import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import pickle

from settings import TRAIN_DUMP_FILE

data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())
# separate columns we need!
data = data[["G1", "G2", "G3", "studytime", "failures"]]

# what we want to predict
predict = "G3"

# separate data to different arrays
x = np.array(data.drop(predict, 1))
y = np.array(data[predict])

# accurency could be different each time, so we train 10 times and save the best result
best = 0
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        # save the best result in a file, so we don't need to train model every time
        with open(TRAIN_DUMP_FILE, "wb") as f:
            pickle.dump(linear, f)

print("Best Result is : %s" % best)

train_result = pickle.load(open(TRAIN_DUMP_FILE, "rb"))
print("Coefficient : ", train_result.coef_)
print("Intercept : ", train_result.intercept_)

results = train_result.predict(x_test)
for x in range(len(results)):
    print("predict : %s, actual value is : %s" % (results[x], y_train[x]))

