import h5py
import numpy as np
import time
import logistic as log

with h5py.File('images_training.h5', 'r') as H:
    trainData = np.copy(H['data'])
with h5py.File('labels_training.h5', 'r') as H:
    trainLabel = np.copy(H['label'])

with h5py.File('images_testing.h5', 'r') as H:
    testData = np.copy(H['data'])
with h5py.File('labels_testing_2000.h5', 'r') as H:
    testLabel = np.copy(H['label'])

testKnowNumber = 2000
testNumber = 5000
trainNumber = 30000


def loadTrainData():
    trainingLength = len(trainData[:trainNumber])
    flattenTrainDate = np.zeros((trainingLength, 784))
    for i, rows in enumerate(trainData[:trainNumber]):
        flattenTrainDate[i] = np.append([], rows)
    print("training data", flattenTrainDate.shape)
    return flattenTrainDate, trainLabel[:trainNumber]


def loadTestData():
    testData_i = testData[:testNumber]
    testVector = np.zeros((testNumber, 784))
    for i, rows in enumerate(testData_i):
        testVector[i] = np.append([], rows)
    print("testing data", testVector.shape)
    return testVector, testLabel[:testKnowNumber]


def logistic_Classy():
    flattenTrainDate, flattenTrainLabel = loadTrainData()
    t1 = time.time()
    all_theta = log.one_vs_all(flattenTrainDate, flattenTrainLabel, 10, 1)
    testVector, testLabel_i = loadTestData()

    test_data = np.zeros((2, 10))
    y_prediction = log.predict_all(testVector, all_theta)
    errorCount = 0
    for i in range(testKnowNumber):
        if y_prediction[i] != testLabel_i[i]:
            test_data[0, testLabel_i[i]] += 1
            errorCount += 1.0
        test_data[1, testLabel_i[i]] += 1

    print("the error number for each label\n", test_data)
    print("the total correct rate is: %.5f" % (1 - (errorCount / float(testKnowNumber))))
    t2 = time.time()
    print("Cost time: %.2fmin, %.4fs." % ((t2 - t1) // 60, (t2 - t1) % 60))

    with h5py.File('predicted_labels.h5', 'w') as H:
        H.create_dataset('label', data=y_prediction[2000:])


logistic_Classy()
