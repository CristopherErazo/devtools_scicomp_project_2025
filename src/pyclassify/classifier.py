from pyclassify.utils import distance
from pyclassify.utils import distance_numpy
from pyclassify.utils import majority_vote
from line_profiler import profile
import numpy as np

class kNN: 
	def __init__(self,k,backhand = 'plain'):
		if not isinstance(k,int):
			raise TypeError(f'Sorry, number of NN should be int, you entered {type(k)}')
		if not backhand == 'plain' and not backhand == 'numpy':
			raise ValueError(f'Sorry, backhand should be either plain or numpy , you entered {backhand}')
		self.k = k
		self.backhand = backhand
		
		if self.backhand == 'plain':
			self.distance = distance
		else:
			self.distance = distance_numpy
	

	@profile
	def _get_k_nearest_neighbors(self,X,Y,x):
		'''
		This method gets the dataset (X,Y) and a new test point x
		and returns a list of the labels of the k-NN of x.
		
		Inputs: 
		X : list(list(float))
		Y : list([0,1])

		Output:
		neighbors : list(int)
		'''

		N_samples = len(Y)
		distances = [self.distance(x,xi) for xi in X]
		idx = sorted(range(N_samples) , key = distances.__getitem__)[:self.k]
		neighbors = [Y[i] for i in idx]
		return neighbors
		
	@profile
	def __call__(self, data ,new_points):
		'''
		This method returns the predicted label for each new 
		point based in the data using the majority vote rule.
		'''
		
		
		X , Y = data
		if self.backhand == 'numpy': 
			X = np.array(X)
			new_points = np.array(new_points)
		predictions = []
		for x in new_points: 
			neighbors = self._get_k_nearest_neighbors(X,Y,x)
			vote = majority_vote(neighbors)
			predictions.append(vote)
		return predictions
	
		
