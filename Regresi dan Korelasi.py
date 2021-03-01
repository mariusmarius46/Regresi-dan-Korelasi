#library-library yang dibutuhkan
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from matplotlib import pyplot as plt


#data untuk variable X untuk Kolom X dan Y untuk kolom Y
X_train = np.array([27,19,15,26,17,25,21,14,27,26,23,18], dtype=float)
y_train = np.array([20,23,18,25,26,24,23,24,20,22,26,28], dtype=float)
X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)


#Create linear regression object
model_reg = LinearRegression()

#train the model using the training sets
model_reg.fit(X_train, y_train)

print('Panjang X_train = {}'.format(len(X_train)))
print('Panjang Y_train = {}'.format(len(y_train)))

#regression coefficients
print('Coefficients b = {}'.format(model_reg.coef_))
print('Constant a ={} '.format(model_reg.intercept_))

#model regresi yang didapat
print('Y = ', model_reg.intercept_ ,'+', model_reg.coef_,'X') 

#prediksi satu data jika nilai X = 28
print('Y = {}'.format(model_reg.predict([[28]])))


X_train = X_train.reshape(-1)
y_train = y_train.reshape(-1)


# Apply the pearsonr()
korelasi = pearsonr(X_train, y_train)
print('Pearsons correlation:', korelasi)



