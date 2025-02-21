import os
import yaml
from line_profiler import profile 
import numpy as np
# Distance function

@profile
def distance(point1,point2): 
	'''
	This function takes two inputs of type: list[float]
	and returns the square euclidean distance between them.
	
	Inputs: 
		- point1 , point2 : list(float) 
	Output:
		dis = || point1 - point2 ||**2 (float)
	
	'''
	if not isinstance(point1,list) or not isinstance(point2,list):
		raise TypeError(f'Sorry, the points must be lists, you entered a {type(point1)} and {type(point2)}')
	
	dis = 0.0
	for i in range(len(point1)): 
		dis += (point1[i] - point2[i])**2
	# dis = dis**(0.5)
	return dis

@profile
def distance_numpy(point1,point2):
	'''
	This function takes two inputs of type: list[float]
	and returns the square euclidean distance between them using numpy
	
	Inputs: 
		- point1 , point2 : list(float) 
	Output:
		dis = || point1 - point2 ||**2 (float)
	
	'''
	if not isinstance(point1,np.ndarray) or not isinstance(point2,np.ndarray):
		raise TypeError(f'Sorry, the points must be np arrays, you entered a {type(point1)} and {type(point2)}')
	
	if not point1.shape == point2.shape:
		raise TypeError(f'The shapes should be the same')
	#dis = np.sum((point1-point2)**2)
	dis = np.linalg.norm(point1-point2)
	return float(dis)

@profile
def majority_vote(neighbors): 
	'''
	This function takes as input a list of the labels of the 
	neighbors of a point and return the label that wins with 
	majority vote
	
	Input: 
		- neighbours: list(int)
	Output:
		- vote
	'''
	if not isinstance(neighbors,list):
		raise TypeError(f'Sorry, the input must be list, you entered a {type(neighbors)}')
	
	k = len(neighbors)
	total = 0
	for lb in neighbors: # We basically count the # of one's
		total += lb
	
	if 2*total > k : # Then there are more ones than zeros
		vote = 1
	else: 
		vote = 0

	return vote


	
def read_config(file):
	filepath = os.path.abspath(f'{file}.yaml')
	print(filepath)
	with open(filepath,	'r') as stream:
		kwargs = yaml.safe_load(stream)
	return kwargs


def read_file(file):
	X , Y = [] , []
	with open(file, 'r') as f: 
		for line in f: 
			values = line.strip().split(',')
			X.append([float(v) for v in values[:-1]])
			label = values[-1]
			if label == 'g' or label == '1': 
				Y.append(1)
			else: 
				Y.append(0)
	data = (X,Y)
	return data

