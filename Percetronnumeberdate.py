from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import numpy as np

digit = datasets.load_digits()
x_train, x_test, y_train, y_test = train_test_split(digit.data, digit.target, train_size= 0.65)
  
p = Perceptron(max_iter=100, eta0=0.001) # epoch, learning rate
p.fit(x_train,y_train)

res = p.predict(x_test)

conf = np.zeros((10,10))
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1
print(conf)

correct = 0
for i in range(10):
    correct += conf[i][i]
accuracy = correct/len(res)
print("정확도는", accuracy*100 , "%입니다.")
    