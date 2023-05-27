import numpy
import torch
import sklearn.datasets as skdata
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from MultiLevelFMM import MLFMM
# print(torch.__version__)
#
# print(torch.cuda.is_available())
#
data = skdata.load_iris()
X = data.data
y = data.target

# X, y = skdata.make_classification(
#     n_samples=2000,  # Number of samples
#     n_features=4,  # Number of features
# )


# ### ### ### ### ### ### ### ### ###

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
# X[np.isnan(X)] = 0
# ### ### ### ### ### ### ### ### ###


# Spliting the Tarin and Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=7)
scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# X_DSEL = scaler.transform(X_DSEL)

clr = MLFMM(theta=.27, gamma=1, mu=.25, no_levels=2, random_state=0)
clr.fit(X_train, y_train)
y_hat =  clr.predict(X_test)
print("accuracy:" , clr.score(X_test,y_test))

X = torch.cuda.FloatTensor(X)
y = torch.cuda.IntTensor(y)
