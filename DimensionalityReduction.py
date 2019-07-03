from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler

class PrincipalComponentAnalysis():

	def __init__(self, dataset):
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
		

	def scaleData(self):
		'''
		Scaling the data so that all different ranges of data gets equal weight
		'''

		self.reshapeData()
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
		principalComponents = pca.fit_transform(X = self.standardizedData)

		return pca.explained_variance_ratio_.sum()

	def getPrincipalComponents_fraction(self, fraction):
		'''
		Returns the principal components based on the given fraction of information
		to be reatined
		'''

		pca = PCA(fraction)
		principalComponents = pca.fit_transform(X = self.standardizedData)

		return principalComponents

	def getPrincipalComponents_noOfComponents(self, noOfComponents):
		'''
		Returns the principal components based on the given nnumber of components
		to be retained
		'''

		pca = PCA(n_components=noOfComponents)
		principalComponents = pca.fit_transform(X = self.standardizedData)

		return principalComponents		


class KernelPCAAlgorithm():

	def __init__(self, dataset, n_jobs):
		'''
		Loading the dataset
		'''
		self.dataset = dataset
		self.n_jobs = n_jobs

	def reshapeData(self):
		'''
		Reshaping the vector from 3D to 2D
		'''
		
		h,x,y = self.dataset.shape
		self.dataset = self.dataset.reshape(x*y,h)
		

	def scaleData(self):
		'''
		Scaling the data so that all different ranges of data gets equal weight
		'''

		self.reshapeData()
		self.standardizedData = StandardScaler().fit_transform(self.dataset)

	def getMinimumComponents(self, fraction):
		'''
		fraction - Fraction of information that needs to be retained

		This method finds the least number of components needed to retain the given
		fraction of information
		'''

		pca = KernelPCA(fraction)
		principalComponents = pca.fit_transform(X = self.standardizedData)

		return self.pca.n_components_

	def getRetainedVariance(self, noOfComponents):
		'''
		noOfComponents - No of components / bands to be used

		This method finds the variance of information retained after using the given
		number of bands
		'''

		pca = KernelPCA(n_components=noOfComponents, kernel="poly")
		principalComponents = pca.fit_transform(X = self.standardizedData)

		return pca.explained_variance_ratio_.sum()

	def getPrincipalComponents_fraction(self, fraction):
		'''
		Returns the principal components based on the given fraction of information
		to be reatined
		'''

		pca = KernelPCA(fraction)
		principalComponents = pca.fit_transform(X = self.standardizedData)

		return principalComponents

	def getPrincipalComponents_noOfComponents(self, noOfComponents, noOfJobs, kernel, solver, alpha, gamma, fit_inv_trans, rem_zero_eigen):
		'''
		Returns the principal components based on the given nnumber of components
		to be retained
		'''

		kpca = KernelPCA(n_components=noOfComponents, kernel=kernel, gamma=gamma, alpha=alpha, fit_inverse_transform=fit_inv_trans, eigen_solver=solver, remove_zero_eig=rem_zero_eigen, n_jobs=noOfJobs)
		principalComponents = kpca.fit_transform(X = self.standardizedData)

		return principalComponents


class LLE():

	def __init__(self, dataset, n_jobs):
		'''
		Loading the dataset
		'''
		self.dataset = dataset
		self.n_jobs = n_jobs

	def reshapeData(self):
		'''
		Reshaping the vector from 3D to 2D
		'''
		
		h,x,y = self.dataset.shape
		self.dataset = self.dataset.reshape(x*y,h)
		

	def scaleData(self):
		'''
		Scaling the data so that all different ranges of data gets equal weight
		'''

		self.reshapeData()
		self.standardizedData = StandardScaler().fit_transform(self.dataset)

	
	def getPrincipalComponents_noOfComponents(self, noOfComponents, noOfNeighbours, noOfJobs, solver, method):
		'''
		Returns the principal components based on the given nnumber of components
		to be retained
		'''

		solver = str(solver)
		method = str(method)
		lle = LocallyLinearEmbedding(n_components=noOfComponents, n_neighbors=noOfNeighbours, n_jobs=self.n_jobs, eigen_solver=solver, method=method)
		principalComponents = lle.fit_transform(X = self.standardizedData)

		return principalComponents
