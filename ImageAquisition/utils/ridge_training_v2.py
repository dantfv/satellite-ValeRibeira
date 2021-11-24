# This is based on code from the Jean et al Github that is modified to work with Python3 and our metrics

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
import sklearn.linear_model as linear_model
from tqdm.notebook import tqdm

class ridge_model():

	def __init__(self, X_set, Y_set, alphas=None, k_folds_train=10, k_folds_repeats=3, scoring='neg_mean_absolute_error', random_seed=7):
		np.random.seed(random_seed)
		print("RANDOM SEED: "+str(random_seed))
		print("K FOLDS TRAIN: "+str(k_folds_train))
		self.X = X_set
		self.Y = Y_set
		self.alpha = None
		self.k_folds_train = k_folds_train
		self.scoring = scoring
		self.cv = RepeatedKFold(n_splits=self.k_folds_train, n_repeats=3, random_state=1)
		self.is_trained = False
		if(alphas is None):
			self.alphas = np.arange(0, 0.1, 0.001)
		else: 
			self.alphas = alphas
		self.model = Ridge()
	
	def __str__(self):
		return "Shape X: "+str(self.X.shape)+"\n" \
		+ "Shape Y: "+str(self.Y.shape)+"\n" \
		+ "Alpha: "+str(self.alpha)+"\n" \
		+ "Model: "+str(self.model)+"\n"
	
	def init_model(self, *, alpha=1):
		return Ridge(alpha=alpha)
	
	def init_cv(self):
		return RepeatedKFold(n_splits=self.k_folds_train, n_repeats=3)

	def train(self):
		print("STARTING TRAINNIG")
		self.is_trained = False
		self.alpha = None
		self.best_r2 = -200
		for a in tqdm(self.alphas):	
			mod = self.init_model(alpha = a)
			cv = self.init_cv()
			#print(mod)
			for train_idx, test_idx in cv.split(self.X):
				X_train, X_test = self.X[train_idx], self.X[test_idx]
				Y_train, Y_test = self.Y[train_idx], self.Y[test_idx]
				#X_train, X_test = self.scale_features(X_train, X_test)
				mod.fit(X_train, Y_train)
				r2 = mod.score(X_test, Y_test)
				#print(r2)
				if r2 > self.best_r2:
					self.alpha = a
					self.best_r2 = r2
					self.model = mod
					print("Melhor resultado encontrado (alpha = " + str(self.alpha) + ", r2 = " + str(self.best_r2) + ")")
					
		self.is_trained = True
		print('melhor alpha encontrado: %f' % self.alpha)
		print('alpha model: %f' % self.model.alpha)
		
	def scale_features(self, X_train, X_test):
		"""
		Scales features using StandardScaler.
		"""
		X_scaler = StandardScaler(with_mean=True, with_std=False)
		X_train = X_scaler.fit_transform(X_train)
		X_test = X_scaler.transform(X_test)
		return X_train, X_test
	
	def predict(self, X):
		assert self.is_trained, "MODELO AINDA NAO TREINADO"
		return self.model.predict(X)
		
	def r2_score(self, Y_real, Y_hat):
		return metrics.r2_score(Y_real, Y_hat)
		
	def score(self, X, Y):
		assert self.is_trained, "MODELO AINDA NAO TREINADO"
		return self.model.score(X, Y)


