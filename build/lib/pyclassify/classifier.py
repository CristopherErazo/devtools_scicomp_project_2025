from pyclassify.utils import distance
from pyclassify.utils import majority_vote

class kNN: 
	def __init__(self,k):
		if not isinstance(k,int):
			raise TypeError(f'Sorry, number of NN should be int, you entered {type(k)}')

		self.k = k
	
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
		distances = [distance(x,xi) for xi in X]
		idx = sorted(range(N_samples) , key = distances.__getitem__)[:self.k]
		neighbors = [Y[i] for i in idx]
		return neighbors

	def __call__(self, data ,new_points):
		'''
		This method returns the predicted label for each new 
		point based in the data using the majority vote rule.
		'''
		X , Y = data
		predictions = []
		for x in new_points: 
			neighbors = self._get_k_nearest_neighbors(X,Y,x)
			vote = majority_vote(neighbors)
			predictions.append(vote)
		return predictions
	
		
