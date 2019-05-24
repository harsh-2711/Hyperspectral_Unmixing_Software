from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCA():

	def __init__(self, dataset):
		'''
		Loading the dataset
		'''
		self.dataset = dataset

	def scaleData(self):
		'''
		Scaling the data so that all different ranges of data gets equal weight
		'''
		self.standardizedData = StandardScaler().fit_transform(self.dataset)

	def getMinimumComponents(self, fraction):
		'''
		fraction - Fraction of information that needs to be retained

		This method finds the least number of components needed to retain the given
		fraction of information
		'''
		pca = PCA(fraction)
		principalComponents = pca.fit_transform(X = self.standardizedData)

		return self.pca.n_components_

	def getRetainedVariance(self, noOfComponents):
		'''
		noOfComponents - No of components / bands to be used

		This method finds the variance of information retained after using the given
		number of bands
		'''
		pca = PCA(n_components=noOfComponents)
		principalComponents = pca.fit_transform(X = standardizedData)

		return pca.explained_variance_ratio_.sum()
