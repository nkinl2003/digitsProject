# digitsProject
필기 인식 프로젝트
미니 프로젝트 : Scikit-learn Toy Dataset 분류
image

Scikit-learn 라이브러리에서는 Toy Datasts(연습용 데이터셋) 와 Real World Datasets(실제 데이터셋) 을 제공한다.

이번에는 Toy datasets 중 다음의 세가지 데이터 셋을 이용하여 여러 분류모델을 사용 해보고, 해당 데이터 마다 어떤 분류모델이 가장 높은 성능 을 보이는지 확인해보겠다.

Optical recognition of handwritten digits dataset : 손글씨 이미지 데이터
Wine recognition dataset: 와인 데이터
Breast cancer wisconsin (diagnostic) dataset : 유방암 데이터
1. 손글씨 이미지 데이터 분류
데이터 불러오기
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score,confusion_matrix, plot_confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
digits = load_digits()    # 데이터 불러오기
print(dir(digits))  # dir : 객체가 어떤 변수와 메소드를 가지고 있는지 나열
['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']
불러온 이미지를 살려보자.

np.random.seed(0)
N = 4
M = 10
fig = plt.figure(figsize=(10, 5))
plt.subplots_adjust(top=1, bottom=0, hspace=0, wspace=0.05)
for i in range(N):
    for j in range(M):
        k = i*M+j
        ax = fig.add_subplot(N, M, k+1)
        ax.imshow(digits.images[k], cmap=plt.cm.bone, interpolation="none")
        ax.grid(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.title(digits.target_names[digits.target[k]])
plt.tight_layout()
plt.show()

위와 같이 손 글씨 이미지 데이터이고 0~9까지의 숫자로 라벨링된 것을 볼 수 있다.
다음으로 DESCR 를 통해 데이터에 대한 설명을 보자

print(digits.DESCR)  # 데이터에 대한 설명
.. _digits_dataset:

Optical recognition of handwritten digits dataset
--------------------------------------------------

**Data Set Characteristics:**

    :Number of Instances: 5620
    :Number of Attributes: 64
    :Attribute Information: 8x8 image of integer pixels in the range 0..16.
    :Missing Attribute Values: None
    :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
    :Date: July; 1998

This is a copy of the test set of the UCI ML hand-written digits datasets
https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

The data set contains images of hand-written digits: 10 classes where
each class refers to a digit.

Preprocessing programs made available by NIST were used to extract
normalized bitmaps of handwritten digits from a preprinted form. From a
total of 43 people, 30 contributed to the training set and different 13
to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of
4x4 and the number of on pixels are counted in each block. This generates
an input matrix of 8x8 where each element is an integer in the range
0..16. This reduces dimensionality and gives invariance to small
distortions.

For info on NIST preprocessing routines, see M. D. Garris, J. L. Blue, G.
T. Candela, D. L. Dimmick, J. Geist, P. J. Grother, S. A. Janet, and C.
L. Wilson, NIST Form-Based Handprint Recognition System, NISTIR 5469,
1994.

.. topic:: References

  - C. Kaynak (1995) Methods of Combining Multiple Classifiers and Their
    Applications to Handwritten Digit Recognition, MSc Thesis, Institute of
    Graduate Studies in Science and Engineering, Bogazici University.
  - E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika.
  - Ken Tang and Ponnuthurai N. Suganthan and Xi Yao and A. Kai Qin.
    Linear dimensionalityreduction using relevance weighted LDA. School of
    Electrical and Electronic Engineering Nanyang Technological University.
    2005.
  - Claudio Gentile. A New Approximate Maximal Margin Classification
    Algorithm. NIPS. 2000.
DESCR 함수를 통해 해당 데이터에 대한 자세한 설명을 살펴보면 다음과 같은 정보들을 얻을수 있다.

총 5620 개의 데이터 -> DESCR 설명 중 오류 -> 총 1797개의 데이터
feature 은 64개의 픽셀값 (8x8 이미지)
10개의 라벨
결측치 없음
digits.target_names   # 라벨 값 
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
digits_data = digits.data      # digits feature데이터 저장 저장
digits_target = digits.target  # digits 의 라벨 저장
print(digits_data.shape)
print(digits_target.shape)
(1797, 64)
(1797,)
여기서 shape 을 통해 데이터를 보면 총 1797개의 데이터가 있음을 알 수 있다.
즉, DESCR를 이용해 불러온 데이터에 대한 설명 중 오류가 있었는 듯 싶다.
shape 을 통해 한번 더 확인을 해봄으로써 데이터에 대한 정확한 이해를 할 수 있다.

데이터 다루기 (training set, test set 나누기)
digits_df = pd.DataFrame(data=digits_data, columns=digits.feature_names)
digits_df.head()
pixel_0_0	pixel_0_1	pixel_0_2	pixel_0_3	pixel_0_4	pixel_0_5	pixel_0_6	pixel_0_7	pixel_1_0	pixel_1_1	...	pixel_6_6	pixel_6_7	pixel_7_0	pixel_7_1	pixel_7_2	pixel_7_3	pixel_7_4	pixel_7_5	pixel_7_6	pixel_7_7
0	0.0	0.0	5.0	13.0	9.0	1.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	6.0	13.0	10.0	0.0	0.0	0.0
1	0.0	0.0	0.0	12.0	13.0	5.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	11.0	16.0	10.0	0.0	0.0
2	0.0	0.0	0.0	4.0	15.0	12.0	0.0	0.0	0.0	0.0	...	5.0	0.0	0.0	0.0	0.0	3.0	11.0	16.0	9.0	0.0
3	0.0	0.0	7.0	15.0	13.0	1.0	0.0	0.0	0.0	8.0	...	9.0	0.0	0.0	0.0	7.0	13.0	13.0	9.0	0.0	0.0
4	0.0	0.0	0.0	1.0	11.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	2.0	16.0	4.0	0.0	0.0
5 rows × 64 columns

digits_df["label"] = digits.target
digits_df.head()
pixel_0_0	pixel_0_1	pixel_0_2	pixel_0_3	pixel_0_4	pixel_0_5	pixel_0_6	pixel_0_7	pixel_1_0	pixel_1_1	...	pixel_6_7	pixel_7_0	pixel_7_1	pixel_7_2	pixel_7_3	pixel_7_4	pixel_7_5	pixel_7_6	pixel_7_7	label
0	0.0	0.0	5.0	13.0	9.0	1.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	6.0	13.0	10.0	0.0	0.0	0.0	0
1	0.0	0.0	0.0	12.0	13.0	5.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	11.0	16.0	10.0	0.0	0.0	1
2	0.0	0.0	0.0	4.0	15.0	12.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	3.0	11.0	16.0	9.0	0.0	2
3	0.0	0.0	7.0	15.0	13.0	1.0	0.0	0.0	0.0	8.0	...	0.0	0.0	0.0	7.0	13.0	13.0	9.0	0.0	0.0	3
4	0.0	0.0	0.0	1.0	11.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	2.0	16.0	4.0	0.0	0.0	4
5 rows × 65 columns

X_train, X_test, y_train, y_test = train_test_split(digits_data, digits_target, test_size = 0.2, random_state = 7)

print(f'X_train 개수 : {len(X_train)}, X_test 개수 :{len(X_test)}')
X_train 개수 : 1437, X_test 개수 :360
모델 학습시키기
Case1. Decision Tree 모델 사용
from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier(random_state = 32)
decision_tree_model.fit(X_train, y_train)
y_pred = decision_tree_model.predict(X_test)

digit_acc={}   # 손글씨 데이터의 정확도 dictionary
digit_acc['Decision Tree'] = accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))
print(digit_acc['Decision Tree'])
              precision    recall  f1-score   support

           0       1.00      0.98      0.99        43
           1       0.81      0.81      0.81        42
           2       0.79      0.82      0.80        40
           3       0.79      0.91      0.85        34
           4       0.83      0.95      0.89        37
           5       0.90      0.96      0.93        28
           6       0.84      0.93      0.88        28
           7       0.96      0.82      0.89        33
           8       0.88      0.65      0.75        43
           9       0.78      0.78      0.78        32

    accuracy                           0.86       360
   macro avg       0.86      0.86      0.86       360
weighted avg       0.86      0.86      0.85       360

0.8555555555555555
Case 2. Random Forest 모델 사용
from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(random_state = 32)
random_forest_model.fit(X_train, y_train)
y_pred = random_forest_model.predict(X_test)

digit_acc['Random Forest'] = accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))
print(digit_acc['Random Forest'])
              precision    recall  f1-score   support

           0       1.00      0.98      0.99        43
           1       0.93      1.00      0.97        42
           2       1.00      1.00      1.00        40
           3       1.00      1.00      1.00        34
           4       0.93      1.00      0.96        37
           5       0.90      0.96      0.93        28
           6       1.00      0.96      0.98        28
           7       0.94      0.97      0.96        33
           8       1.00      0.84      0.91        43
           9       0.94      0.94      0.94        32

    accuracy                           0.96       360
   macro avg       0.96      0.96      0.96       360
weighted avg       0.97      0.96      0.96       360

0.9638888888888889
Case 3. SVM 모델 사용
from sklearn import svm

svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

digit_acc['SVM'] = accuracy_score(y_test, y_pred)


print(classification_report(y_test, y_pred))
print(digit_acc['SVM'])
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        43
           1       0.95      1.00      0.98        42
           2       1.00      1.00      1.00        40
           3       1.00      1.00      1.00        34
           4       1.00      1.00      1.00        37
           5       0.93      1.00      0.97        28
           6       1.00      1.00      1.00        28
           7       1.00      1.00      1.00        33
           8       1.00      0.93      0.96        43
           9       1.00      0.97      0.98        32

    accuracy                           0.99       360
   macro avg       0.99      0.99      0.99       360
weighted avg       0.99      0.99      0.99       360

0.9888888888888889
Case 4. SGD Classifier 모델 사용
from sklearn.linear_model import SGDClassifier

sgd_model = SGDClassifier()
sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)

digit_acc['SGD Classifier'] = accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))
print(digit_acc['SGD Classifier'])
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        43
           1       0.93      0.90      0.92        42
           2       0.98      1.00      0.99        40
           3       0.97      0.88      0.92        34
           4       0.97      0.97      0.97        37
           5       0.80      1.00      0.89        28
           6       0.96      0.96      0.96        28
           7       0.97      0.94      0.95        33
           8       0.97      0.84      0.90        43
           9       0.86      0.94      0.90        32

    accuracy                           0.94       360
   macro avg       0.94      0.94      0.94       360
weighted avg       0.95      0.94      0.94       360

0.9416666666666667
Case 5. Logistic Regression
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

digit_acc['Logistic Regression'] = accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))
print(digit_acc['Logistic Regression'])
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        43
           1       0.95      0.95      0.95        42
           2       0.98      1.00      0.99        40
           3       0.94      0.97      0.96        34
           4       0.97      1.00      0.99        37
           5       0.82      0.96      0.89        28
           6       1.00      0.96      0.98        28
           7       0.97      0.97      0.97        33
           8       0.92      0.81      0.86        43
           9       0.97      0.91      0.94        32

    accuracy                           0.95       360
   macro avg       0.95      0.95      0.95       360
weighted avg       0.95      0.95      0.95       360

0.9527777777777777
/opt/conda/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
모델 성능 평가
모델의 성능을 평가하는 척도는 손글씨의 숫자가 뭔지 예측하는 것이므로 전체 개수 중 맞은 것의 개수의 수치인 정확도(accuracy) 를 이용하겠다.

for i in digit_acc.items():
    print("{0:<20} : {1}".format(i[0],i[1]))
Decision Tree        : 0.8555555555555555
Random Forest        : 0.9638888888888889
SVM                  : 0.9888888888888889
SGD Classifier       : 0.9416666666666667
Logistic Regression  : 0.9527777777777777
손글씨 이미지 데이터는 SVM 모델을 사용했을때 모델의 성능이 가장 높게 나온 것을 확인 할 수 있다.

손글씨 이미지 데이터의 SVM 모델을 사용했을 때 정확도(accuracy) 는 약 98.89% 이다.
