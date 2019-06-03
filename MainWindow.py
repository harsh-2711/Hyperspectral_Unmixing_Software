import sys
import os

from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QPushButton, QTextEdit, QListWidget, QProgressBar, QLabel

from osgeo import gdal
from numpy import genfromtxt

from PCA import PrincipalComponentAnalysis
from NMF import NonNegativeMatrixFactorisation
from Threads import ValidationThread
from ErrorHandler import StdErrHandler

import multiprocessing as mp

import matplotlib.pyplot as plt


path = os.path.dirname(__file__)
qtCreatorFile = "MainWindow.ui" 

Ui_MainWindow, QtBaseClass = uic.loadUiType(path + qtCreatorFile)

class Software(QMainWindow, Ui_MainWindow):

	def __init__(self):
		'''
		Initializing software
		'''

		super(Software, self).__init__()
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		self.title = "Hyperspectral Unmixing Toolbox"
		self.OUTPUT_FILENAME = "Data.mat"
		self.file = ""

		self.initUI()

	def initUI(self):
		'''
		Initializing UI components and their respective listeners
		'''

		self.setWindowTitle(self.title)

		# Input Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 30)

		# Input browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,30)
		self.input_browse.clicked.connect(self.on_click_input)

		# Input text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,35,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 91)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,91)
		self.output_browse.clicked.connect(self.on_click_output)

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,95,402,21)
		self.output_text.setText(os.getcwd())

		# No of components Label
		self.components_label = QLabel("Components", self)
		self.components_label.move(20, 150)

		# Components text field
		self.components = QTextEdit(self)
		self.components.setGeometry(142,155,502,21)

		# Jobs Label
		self.jobs_label = QLabel("No of jobs", self)
		self.jobs_label.move(20, 210)

		# Jobs text field
		self.jobs = QTextEdit(self)
		self.jobs.setGeometry(142,215,502,21)

		# OK button
		self.OK = QPushButton("OK", self)
		self.OK.move(230, 250)
		self.OK.clicked.connect(self.on_click_OK)

		# Cancel button
		self.cancel = QPushButton("Cancel", self)
		self.cancel.move(380, 250)
		self.cancel.clicked.connect(self.on_click_cancel)

		# Logs entry
		self.logs = QListWidget(self)
		self.logs.setGeometry(10, 300, 640, 130)

		# Progress Bar
		self.progress = QProgressBar(self)
		self.progress.setGeometry(10, 450, 640, 20)

		self.show()

	@pyqtSlot()
	def on_click_input(self):
		'''
		On click listener for input_browse button
		'''

		self.InputBrowse()

	def InputBrowse(self):
		'''
		Opens Browse Files dialog box for selecting input dataset
		'''

		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(self,"Select Dataset", "","All Files (*);;Matlab Files (*.mat)", options=options)
		self.file = fileName
		if fileName:
			self.input_text.setText(fileName.split('/')[-1])

	@pyqtSlot()
	def on_click_output(self):
		'''
		On click listener for output_browse button
		'''

		self.OutputBrowse()

	def OutputBrowse(self):
		'''
		Opens Browse Files dialog box for selecting target file for writing output
		'''

		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		folderName = str(QFileDialog.getExistingDirectory(self, "Select Directory", options=options))
		if folderName:
			self.output_text.setText(folderName)

	@pyqtSlot()
	def on_click_OK(self):
		'''
		On click listener for OK button
		'''

		self.validation = ValidationThread()
		self.validation.startThread.connect(self.validate)
		self.validation.startProgress.connect(self.setProgressBar)
		self.validation.start()

	@pyqtSlot()
	def on_click_cancel(self):
		'''
		On click listener for Cancel button
		'''

		self.input_text.setText("")
		self.output_text.setText("")
		self.components.setText("")
		self.jobs.setText("")
		self.progress.setValue(0)
		self.logs.clear()

	def setProgressBar(self, switch):
		'''
		switch - Boolean 

		Switches the progress bar from busy to stop and vice versa based on the
		value of switch
		'''
		if switch:
			self.progress.setRange(0,0)
		else:
			self.progress.setRange(0,1)

	def validate(self):
		'''
		Parent function for validating all the input fields
		'''

		# Suppressing printing of errors using GDAL lib
		gdal.UseExceptions()
		gdal.PushErrorHandler('CPLQuietErrorHandler')

		filename = self.input_text.toPlainText()
		foldername = self.output_text.toPlainText()
		selectedComponents = self.components.toPlainText()
		n_jobs = self.jobs.toPlainText()

		# Validating dataset path
		self.dataExists = self.validateInputFile(self.file)

		# Validating output folder path
		if self.dataExists:
			self.outputFolderExists = self.validateOutputFolder(foldername)

		# Validating number of components
		if self.dataExists and self.outputFolderExists:
			self.trueComponents = self.validateComponents(selectedComponents)

		# Validating number of jobs
		if self.dataExists and self.outputFolderExists and self.trueComponents:
			self.enoughProcs = self.validateJobs(n_jobs)

		# Start PCA if everything's good
		if self.dataExists and self.outputFolderExists and self.trueComponents and self.enoughProcs:
			self.logs.addItem(f'Starting Principal Component Analysis for getting top {self.components.toPlainText()} bands')
			# self.startPCA(selectedComponents)
			self.startNMF(selectedComponents)
		
	def validateInputFile(self, filename):
		'''
		Validates the dataset path and loads the dataset if path exists
		'''

		if filename:
			try:
				self.dataset = gdal.Open(filename, gdal.GA_ReadOnly)
				self.logs.addItem("Dataset imported successfully")
				return True
			except:
				self.logs.addItem(gdal.GetLastErrorMsg())
				self.logs.addItem('Use command line argument gdalinfo --formats to get more insights')
				return False
		else:
			self.logs.addItem("Please provide path to dataset")
			return False

	def validateOutputFolder(self, foldername):
		'''
		Validates the existence of output folder where outfile file will be 
		created after analysis
		'''

		if foldername:
			if os.path.isdir(foldername):
				return True
		self.logs.addItem("Please provide a valid directory to save output file")
		return False

	def validateComponents(self, selectedComponents):
		'''
		Validates the number of components w.r.t. the input dataset
		'''

		totalComponents = self.dataset.RasterCount
		if selectedComponents.isdigit():
			if (int)(selectedComponents) > 0 and (int)(selectedComponents) <= totalComponents:
				return True
		self.logs.addItem(f'Incorrect number of bands... Max possible number of bands are {totalComponents}')
		return False

	def validateJobs(self, n_jobs):
		'''
		Validates the number of jobs desired as per processors available
		'''
		n_processors = mp.cpu_count()
		if n_jobs.isdigit():
			if (int)(n_jobs) > 0 and (int)(n_jobs) <= n_processors:
				return True
		self.logs.addItem(f'Number of jobs must be greater than 0 and less than {n_processors}')
		return False
	
	
	def startPCA(self,selectedComponents):
		'''
		Main function for PCA
		'''

		self.datasetAsArray = self.dataset.ReadAsArray()
		pca = PrincipalComponentAnalysis(self.datasetAsArray)
		pca.scaleData()
		pca_data = pca.getPrincipalComponents_noOfComponents((int)(self.components.toPlainText()))
		retainedVariance = pca.getRetainedVariance((int)(self.components.toPlainText()))
		self.logs.addItem("Analysis completed")
		self.logs.addItem(f'Retained Variance: {retainedVariance}')
		self.logs.addItem("Generating Output file")
		pca_data.tofile(self.OUTPUT_FILENAME, ",")
		self.logs.addItem("Output file generated")
		self.setProgressBar(False)
		
		''' To plot the points after PCA '''
		if (int)(selectedComponents) == 1:
			self.plot1DGraph(pca_data)

		elif (int)(selectedComponents) == 2:
			self.plot2DGraph(pca_data)

		elif (int)(selectedComponents) == 3:
			self.plot3DGraph(pca_data)

		else:
			self.logs.addItem('Due to high dimentionality, graph could not be plotted')


	def startNMF(self,selectedComponents):
		'''
		Main function for NMF
		'''

		self.datasetAsArray = self.dataset.ReadAsArray()
		nmf = NonNegativeMatrixFactorisation(self.datasetAsArray)
		nmf.scaleData()
		nmf_data = nmf.getReducedComponents_noOfComponents((int)(self.components.toPlainText()))
		error = nmf.errorFactor((int)(self.components.toPlainText()))
		self.logs.addItem("Analysis completed")
		self.logs.addItem(f'RMS Error: {error}')
		self.logs.addItem("Generating Output file")
		nmf_data = nmf.denormalizeData(nmf_data)
		nmf_data.tofile(self.OUTPUT_FILENAME, ",")
		self.logs.addItem("Output file generated")
		self.setProgressBar(False)
		
		''' To plot the points after NMF '''
		if (int)(selectedComponents) == 1:
			self.plot1DGraph(nmf_data)

		elif (int)(selectedComponents) == 2:
			self.plot2DGraph(nmf_data)

		elif (int)(selectedComponents) == 3:
			self.plot3DGraph(nmf_data)

		else:
			self.logs.addItem('Due to high dimentionality, graph could not be plotted')


	def plot1DGraph(self,data):

		x = data[:,0]
		y = np.zeros((len(x),), dtype=np.int)
		plt.close('all')
		fig1 = plt.figure()
		pltData = [x,y]
		plt.scatter(pltData[0],pltData[1])
		
		xAxisLine = ((min(pltData[0]), max(pltData[0])), (0, 0))
		plt.plot(xAxisLine[0], xAxisLine[1], 'r')
		# yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1])))
		# plt.plot(yAxisLine[0], yAxisLine[1], 'r')

		plt.xlabel("comp 1") 
		plt.title("1D plot")
		plt.show()

	def plot2DGraph(self,data):

		x = data[:,0]
		y = data[:,1]
		plt.close('all')
		fig1 = plt.figure()
		pltData = [x,y]
		plt.scatter(pltData[0],pltData[1])
		
		xAxisLine = ((min(pltData[0]), max(pltData[0])), (0, 0))
		plt.plot(xAxisLine[0], xAxisLine[1], 'r')
		yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1])))
		plt.plot(yAxisLine[0], yAxisLine[1], 'r')

		plt.xlabel("comp 1") 
		plt.ylabel("comp 2")
		plt.title("2D plot")
		plt.show()	

	def plot3DGraph(self,data):

		x = data[:,0]
		y = data[:,1]
		z = data[:,2]
		plt.close('all')
		fig1 = plt.figure()
		ax = Axes3D(fig1)
		pltData = [x,y,z]
		ax.scatter(pltData[0],pltData[1],pltData[2])
		
		xAxisLine = ((min(pltData[0]), max(pltData[0])), (0, 0), (0,0))
		ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
		yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1])), (0,0))
		ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
		zAxisLine = ((0, 0), (0,0), (min(pltData[2]), max(pltData[2])))
		ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

		ax.set_xlabel("comp 1") 
		ax.set_ylabel("comp 2")
		ax.set_zlabel("comp 3")
		ax.set_title("3D plot")
		plt.show()

	def writeError(self, err_msg):
		'''
		This method receives input from stderr as PyQtSlot and prints it in the 
		logs section
		'''

		self.logs.addItem(err_msg)


if __name__ == "__main__":

	app = QApplication(sys.argv)
	window = Software()
	window.show()

	# Adding error handler
	std_err_handler = StdErrHandler()
	sys.stderr = std_err_handler
	std_err_handler.err_msg.connect(window.writeError)

	sys.exit(app.exec_())
