import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import scipy.special

style.use('ggplot')

data = pd.read_csv('DataSet_Thaba_Classification.csv', sep=';')
print(data.head())

print(data.info())

for i in data.columns:
    if (data[i].isnull()).any():
        data[i].replace(np.nan, data[i].mean(), inplace=True)

for i in data.columns:
    print(data[i].isnull().value_counts())

print(data.info())


def cat2num(arr):
    di = dict.fromkeys(arr)
    val = np.arange(0, len(di) - 1)
    i = 0
    for j in di:
        di[j] = i
        i += 1
    return di


cat2numdict = {'Motherhole': np.nan, 'HoleType': np.nan, 'Stratigraphy': np.nan}
dict1 = cat2num(data['Motherhole'])
dict2 = cat2num(data['HoleType'])
dict3 = cat2num(data['Stratigraphy'])
cat2numdict['Motherhole'] = dict1
cat2numdict['HoleType'] = dict2
cat2numdict['Stratigraphy'] = dict3
data.replace(cat2numdict, inplace=True)
print(data.info())

X = np.array(data.drop(['ProjectCode', 'BH_ID', 'Date', 'Stratigraphy'], axis=1)).astype('float32')
y = np.array(data['Stratigraphy'])


def StandardScalar(arr):
    try:
        for i in range(arr.shape[1]):
            mean = arr[:, i].mean()
            std = arr[:, i].std()
            arr[:, i] = (arr[:, i] - mean) / std

    except IndexError:
        mean = arr.mean()
        std = arr.std()
        arr = (arr - mean) / std

    return arr


print(X[:, 0].mean(), X[:, 0].std())
X = StandardScalar(X)
print(X[:, 0].mean(), X[:, 0].std())
print(len(np.unique(y)))


def train_test_split(X, y, testing_size=0.2):
    total_rows_no = X.shape[0]
    testin_rows_no = int(testing_size * total_rows_no)
    rand_row_no = np.random.randint(0, total_rows_no, testin_rows_no)

    X_train = np.array(X[rand_row_no])
    X_test = np.delete(X, rand_row_no, axis=0)

    y_train = np.array(y[rand_row_no])
    y_test = np.delete(y, rand_row_no, axis=0)

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = train_test_split(X, y, testing_size=0.2)

'''fig, ax = plt.subplots(X.shape[1], 1)
for i in range(X.shape[1]):
    ax[i] = plt.scatter(X[:, i], y)
    plt.show()'''


class DenseNeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.001, epochs=1):
        self.inodes = input_nodes
        self.onodes = output_nodes
        self.hnodes = hidden_nodes
        self.lr = learning_rate
        self.epochs = epochs
        self.wih = np.random.randn(self.hnodes, self.inodes) * 0.1
        self.who = np.random.randn(self.onodes, self.hnodes) * 0.1
        self.activation_sigmoid = lambda x: scipy.special.expit(x)
        self.activation_softmax = lambda x: scipy.special.softmax(x)
        self.loss = 0.0

    def forwardprop(self, sample_input):
        sample_input = sample_input.reshape(-1, 1)

        hidden_inputs = self.wih.dot(sample_input)
        hidden_outputs = self.activation_sigmoid(hidden_inputs)

        classifier_inputs = self.who.dot(hidden_outputs)
        classifier_outputs = self.activation_softmax(classifier_inputs)

        self.outputs = classifier_outputs
        self.hidden_outputs = hidden_outputs

    def CategoricalCrossEntropy(self, classifier_outputs, sample_output):
        one_hot_encoded_matrix = np.zeros((self.onodes, 1)) + 0.01
        one_hot_encoded_matrix[sample_output] = 0.99

        self.error = np.sum(one_hot_encoded_matrix * np.log(classifier_outputs))

        self.loss += self.error

    def Backprop(self, sample_input, sample_output):
        sample_input = sample_input.reshape(-1, 1)
        one_hot_encoded_matrix = np.zeros((self.onodes, 1)) + 0.01
        one_hot_encoded_matrix[sample_output] = 0.99

        dl_dwo = np.dot((self.outputs - one_hot_encoded_matrix), self.hidden_outputs.transpose())

        dzo_dah = self.who
        dl_dah = np.dot(dzo_dah.transpose(), (self.outputs - one_hot_encoded_matrix))
        dah_dzh = self.hidden_outputs
        dzh_dwh = sample_input
        dl_dwh = np.dot(dah_dzh * dl_dah, dzh_dwh.transpose())

        self.who -= self.lr * dl_dwo
        self.wih -= self.lr * dl_dwh

    def fit(self, X_train, y_train):
        for j in range(self.epochs):
            correct = 0
            for i in range(len(X_train)):
                self.forwardprop(X_train[i])
                if np.argmax(self.outputs) == y_train[i]:
                    correct += 1
                self.CategoricalCrossEntropy(self.outputs, y_train[0])
                self.Backprop(X_train[i], y_train[i])
            accuracy = correct / X_train.shape[0]
            print(f'Epoch: {j + 1} / {self.epochs} \n loss: {-(self.loss / X_train.shape[0])} accuracy = {accuracy}')
            self.loss = 0

    def fit1(self, X_train, y_train, batch_size):
        for i in range(self.epochs):
            j = 0
            correct = 0
            e = X_train.shape[0] // batch_size
            for k in range(X_train.shape[0] // batch_size):
                input_batch = X_train[j:j + batch_size]
                output_batch = y_train[j:j + batch_size]
                for l in range(batch_size):
                    self.forwardprop(input_batch[l])
                    self.CategoricalCrossEntropy(self.outputs, output_batch[l])
                    if np.argmax(self.outputs) == output_batch[l]:
                        correct += 1
                self.Backprop(input_batch[-1], output_batch[-1])
                j += batch_size
            acc = correct / X_train.shape[0]
            print(f'Epoch: {i + 1}/{self.epochs}\n loss: {-(self.loss / X_train.shape[0])} accuracy: {acc} ')
            self.loss = 0


print(X_train.shape[0])
model = DenseNeuralNetwork(19, 100, 15, learning_rate=0.3, epochs=10)
model.fit(X_train, y_train)

correct = 0
for i in range(X_test.shape[0]):
    model.forwardprop(X_test[i])
    if np.argmax(model.outputs) == y_test[i]:
        correct += 1
acc = correct / X_test.shape[0]
print(acc)

'''from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers

model = keras.Sequential(
    [
        layers.Dense(19, activation='relu', kernel_regularizer=regularizers.l2()),
        layers.Dense(300, activation='relu', kernel_regularizer=regularizers.l2()),
        layers.Dense(300, activation='sigmoid', kernel_regularizer=regularizers.l2()),
        layers.Dropout(0.7),
        layers.Dense(15)
    ]
                        )

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

#print(model.summary())

model.fit(X_train, y_train, batch_size=32, epochs=10)
model.evaluate(X_test, y_test, batch_size=32)'''
