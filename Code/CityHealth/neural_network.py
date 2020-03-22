

# Train an articial neural network for predict diabetes risk in each tract
import Code.cityhealth.CityHealth as CH

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


X_train, X_test, y_train, y_test = CH.data_split(data, y_var="Diabetes", test_size=0.3)

# define base model
def baseline_model():
	# create model
    n_vars = 21
	model = Sequential()
	model.add(Dense(n_vars, input_dim=n_vars, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=1)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
