def startUCLS(self, pca_data, nfindr_data):
	'''
	Main function to run UCLS
	'''

	self.logs.addItem("Initiating UCLS algorithm")
	self.UCLS_data = LLM.ATGP(np.transpose(pca_data), nfindr_data)
	self.logs.addItem("Analysis completed")
	self.logs.addItem("Generating output file")
	self.writeData("UCLS_", self.UCLS_data)
	self.logs.addItem(f"Output File UCLS_{self.OUTPUT_FILENAME} generated")
	self.setProgressBar(False)

def startUCLS(self, pca_data, nfindr_data):
	'''
	Main function to run NNLS
	'''

	self.logs.addItem("Initiating NNLS algorithm")
	self.NNLS_data = LLM.ATGP(np.transpose(pca_data), nfindr_data)
	self.logs.addItem("Analysis completed")
	self.logs.addItem("Generating output file")
	self.writeData("NNLS_", self.NNLS_data)
	self.logs.addItem(f"Output File NNLS_{self.OUTPUT_FILENAME} generated")
	self.setProgressBar(False)