from sklearn.metrics import recall_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_score, roc_curve, auc, roc_auc_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, advanced_activations
from sklearn.model_selection import train_test_split
import numpy
import pandas
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


def Neural_net(x_train_res, y_train_res, method, feature_count):
    global test
    global test_labels
    global x_val
    global y_val
    model = Sequential()
    model.add(Dense(50, kernel_initializer='uniform', input_dim=feature_count))
    model.add(advanced_activations.ELU(alpha=1.0))
    model.add(Dropout(0.8))
    model.add(Dense(30, kernel_initializer='uniform'))
    model.add(advanced_activations.ELU(alpha=1.0))
    model.add(Dropout(0.9))
    model.add(Dense(15, kernel_initializer='uniform', activation="relu"))
    # model.add(Dense(15, kernel_initializer='uniform'))
    # model.add(advanced_activations.ELU(alpha=1.0))
    # model.add(Dense(10, init='uniform', activation=''))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

    model.fit(x_train_res, y_train_res, epochs=10, batch_size=1000, validation_data=(x_val, y_val))
    # output = model.predict_classes(x_val)
    # output = [i[0] for i in output]
    # output = numpy.array(output)
    # print(output[y_val == 1])
    # print(sum(output[y_val == 1]))
    # print(sum(output))
    # print(sum(y_val))
    # print(recall_score(y_val, output))
    # print(precision_score(y_val, output))

    # scores = model.evaluate(train_data, train_labels)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("complete")

    ##Computing false and true positive rates
    output = model.predict_classes(test)
    output = [i[0] for i in output]
    output = numpy.array(output)
    print(output[test_labels == 1])
    print(sum(output[test_labels == 1]))
    print(sum(output))
    print(sum(test_labels))
    print(recall_score(test_labels, output))
    print(precision_score(test_labels, output))
    fpr, tpr, _ = roc_curve(output, test_labels, drop_intermediate=False)
    if sum(numpy.isnan(tpr)) == 0:
        roc_acc = roc_auc_score(output, test_labels)
        roc_acc = "Accuracy: " + str(round(roc_acc, 3))
    else:
        roc_acc = "Accuracy: 0%"

    plt.figure()
    ##Adding the ROC
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
    ##Random FPR and TPR
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    ##Title and label
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Neural Network - (' + method + ")" + str(roc_acc))
    # plt.show()
    plt.savefig(fname=method + ".png", pad_inches=0.2)


if __name__ == "__main__":
    global test, test_labels, x_val, y_val
    filepath = "creditcard.csv"
    data = pandas.read_csv(filepath)
    labels = numpy.array(data.Class)
    data.pop("Class")
    feature_count = data.shape[1]
    train_data, test, train_labels, test_labels = train_test_split(data, labels, train_size=200000)
    x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=.1, random_state=12)
    Neural_net(x_train, y_train, "Rawdata", feature_count)

    # columns = numpy.asarray(data.columns)
    # k = len(numpy.unique(labels))

    data1 = data[["V14", "V10", "V12", "V17", "V11"]]

    train_data, test, train_labels, test_labels = train_test_split(data1, labels, train_size=200000)
    x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=.1, random_state=12)
    Neural_net(x_train, y_train, "V14,V10,V12,V17,V11", 5)

    sm = SMOTE(random_state=12, ratio=1.0)
    x_train_res, y_train_res = sm.fit_sample(x_train, y_train)
    Neural_net(x_train_res, y_train_res, "SMOTE", 5)

    rus = RandomUnderSampler(random_state=12)
    x_train_res, y_train_res = rus.fit_sample(x_train, y_train)
    Neural_net(x_train_res, y_train_res, "RandomUnderSampler", 5)
