from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np

class NonNegativeMatrixFactorisation():

	def __init__(self,dataset):
		'''
		Loading the dataset
		'''
		self.dataset = dataset

	def reshapeData(self):
		'''
		Reshaping the vector from 3D to 2D
		'''
		
		h,x,y = self.dataset.shape
		self.dataset = self.dataset.reshape(x*y,h)


	def normalizeData(self):
		'''
		Normalizing dataset
		'''

		# self.dataset = preprocessing.normalize(self.dataset)
		self.minimum = np.min(self.standardizedData)
		self.maximum = np.max(self.standardizedData)
		self.normalizedData = (self.standardizedData - self.minimum) / (self.maximum - self.minimum)


	def printDataset(self):
		print(self.dataset)
		print("done")


	def scaleData(self):
		'''
		Scaling the data so that all different ranges of data gets equal weight
		'''

		self.reshapeData()
		self.standardizedData = StandardScaler(with_mean = False).fit_transform(self.dataset)
		self.normalizeData()


	def denormalizeData(self,dataset):
		self.denormData = (dataset * (self.maximum - self.minimum)) + self.minimum
		return self.denormData


	def getMinimumComponents(self, fraction):
		'''
		fraction - Fraction of information that needs to be retained

		This method finds the least number of components needed to retain the given
		fraction of information
		'''

		nmf = NMF(fraction)
		principalComponents = nmf.fit_transform(X = self.normalizedData)

		return self.nmf.n_components_

	def getRetainedVariance(self, noOfComponents):
		'''
		noOfComponents - No of components / bands to be used

		This method finds the variance of information retained after using the given
		number of bands
		'''

		nmf = NMF(n_components=noOfComponents)
		self.reducedComponents = nmf.fit_transform(X = self.normalizedData)

		return nmf.explained_variance_ratio_.sum()

	def errorFactor(self, noOfComponents):
		'''
		Calculates the difference between the input values and the reduced values
		'''
		nmf = NMF(n_components=noOfComponents)
		W = nmf.fit_transform(X = self.normalizedData)
		H = nmf.components_
		error = ((self.normalizedData - np.matmul(W,H))**2).sum()
		error = error**0.5

		return error



	def getReducedComponents_fraction(self, fraction):
		'''
		Returns the principal components based on the given fraction of information
		to be reatined
		'''

		nmf = NMF(fraction)
		self.reducedComponents = nmf.fit_transform(X = self.normalizedData)

		return self.reducedComponents

	def getReducedComponents_noOfComponents(self, noOfComponents):
		'''
		Returns the principal components based on the given nnumber of components
		to be retained
		'''

		nmf = NMF(n_components=noOfComponents)
		self.reducedComponents = nmf.fit_transform(X = self.normalizedData)
		
		return self.reducedComponents				
