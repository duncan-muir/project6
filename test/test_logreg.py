import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""

def test_updates():
	"""
	Test to check gradients and loss are calculated as expected, and that training losses are reasonable
	"""
	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen',
				  'Plain chest X-ray (procedure)', 'Low Density Lipoprotein Cholesterol',
				  'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	log_model = logreg.LogisticRegression(num_feats=6, max_iter=10000, tol=0.00001, learning_rate=0.01, batch_size=200,
										  rand=np.random.RandomState(4))

	# Check that your gradient is being calculated correctly
	assert np.allclose(log_model.calculate_gradient(X_train[:1], y_train[:1]),
						np.array([-0.01088303, -0.05233874, 0.03182683, -0, -0, 0.04925154]), .0001)


	
	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training

	assert np.isclose(log_model.loss_function(X_train[:1], y_train[:1]), 0.07348, .001)

	log_model.train_model(X_train, y_train, X_val, y_val)

	assert np.all(np.array(log_model.loss_history_train[-10:]) < 0.4)
	assert np.all(np.array(log_model.loss_history_val[-10:]) < 0.4)


def test_predict():
	"""
	Check that parameter updates are behaving as expected
	"""


	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen',
				  'Plain chest X-ray (procedure)', 'Low Density Lipoprotein Cholesterol',
				  'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	log_model = logreg.LogisticRegression(num_feats=6, max_iter=2, tol=0.00001, learning_rate=0.01,
										  batch_size=len(X_train) * 2,
										  rand=np.random.RandomState(4))

	assert np.allclose(log_model.W,
					   [0.05056171, 0.49995133, -0.99590893, 0.69359851, -0.41830152, -1.58457724, -0.64770677], .001)
	old_w = log_model.W

	# train 1 iteration

	X_train_expanded = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
	grad = log_model.calculate_gradient(X_train_expanded, y_train)

	log_model.train_model(X_train, y_train, X_val, y_val)

	new_w = log_model.W

	# compare with manual grad calculation
	assert not np.array_equal(old_w, new_w)
	assert np.allclose(old_w - grad * log_model.lr, new_w, .0001)


def test_accuracy():
	"""
	Check that trained model accuracy is reasonable (>> .50)
	"""
	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen',
				  'Plain chest X-ray (procedure)', 'Low Density Lipoprotein Cholesterol',
				  'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	log_model = logreg.LogisticRegression(num_feats=6, max_iter=10000, tol=0.00001, learning_rate=0.01, batch_size=200,
										  rand=np.random.RandomState(4))


	log_model.train_model(X_train, y_train, X_val, y_val)

	assert accuracy_score([1 if pred >= 0.5 else 0 for pred in log_model.make_prediction(X_train)], y_train) > .80
	assert accuracy_score([1 if pred >= 0.5 else 0 for pred in log_model.make_prediction(X_val)], y_val)  > .80
