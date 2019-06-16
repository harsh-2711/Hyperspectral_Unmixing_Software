import sys
import os

from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot

from functools import partial

from mpl_toolkits.mplot3d import Axes3D

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QPushButton, QTextEdit, QListWidget, QProgressBar, QLabel, QAction
from PyQt5 import QtGui

from osgeo import gdal

import numpy as np
from numpy import genfromtxt
import csv

from DimensionalityReduction import PrincipalComponentAnalysis, KernelPCAAlgorithm, LLE
from NMF import NonNegativeMatrixFactorisation
from nfindr import NFindrModule
from sunsal import SUNSALModule
import vd
from Modules.Material_Count import vd_int

from Modules.End_Member_Extraction import eea
from Modules.Linear_Unmixing import sparse, LMM

from Threads import ValidationThread
from threading import Thread
import subprocess
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
		self.OUTPUT_FILENAME = "Data.csv"
		self.file = ""
		self.currentAlgo = "PCA"

		# Initialising Menu Bar
		self.initMenu()

		self.initUI()


	def initMenu(self):
		'''
		Initialises the menu bar in the software for easy navigation
		'''

		menubar = self.menuBar()
		file = menubar.addMenu("File")

		new = QAction("New", self)
		new.setShortcut("Ctrl+N")
		file.addAction(new)

		save = QAction("Save", self)
		save.setShortcut("Ctrl+S")
		file.addAction(save)

		''' Algorithm Section '''

		# Dimensionality Reduction
		dimReduction = menubar.addMenu("Dimensionality Reduction")

		pcaMenu = QAction("PCA", self)
		dimReduction.addAction(pcaMenu) 
		pcaMenu.triggered.connect(partial(self.changeCurrentAlgo, "PCA"))

		nmf = QAction("NMF", self)
		dimReduction.addAction(nmf)
		nmf.triggered.connect(partial(self.changeCurrentAlgo, "NMF"))

		kerPCA = QAction("Kernel PCA", self)
		dimReduction.addAction(kerPCA)
		kerPCA.triggered.connect(partial(self.changeCurrentAlgo, "KerPCA"))

		lda = QAction("FDA", self)
		dimReduction.addAction(lda) 

		lle = QAction("LLE", self)
		dimReduction.addAction(lle)
		lle.triggered.connect(partial(self.changeCurrentAlgo, "LLE"))

		# Material Count
		mc = menubar.addMenu("Material Count")

		virDim = QAction("Virtual Dimension", self)
		mc.addAction(virDim)

		hysime = QAction("Hysime", self)
		mc.addAction(hysime)

		hfcvd = QAction("HfcVd", self)
		mc.addAction(hfcvd)
		hfcvd.triggered.connect(partial(self.changeCurrentAlgo, "HfcVd"))

		# End Member Extraction
		eme = menubar.addMenu("End Member Extraction")

		nFinder = QAction("N-Finder", self)
		eme.addAction(nFinder)
		nFinder.triggered.connect(partial(self.changeCurrentAlgo, "NFinder"))

		atgp = QAction("ATGP", self)
		eme.addAction(atgp)
		atgp.triggered.connect(partial(self.changeCurrentAlgo, "ATGP"))

		ppi = QAction("PPI", self)
		eme.addAction(ppi)
		ppi.triggered.connect(partial(self.changeCurrentAlgo, "PPI"))

		sisal = QAction("SISAL", self)
		eme.addAction(sisal)
		sisal.triggered.connect(partial(self.changeCurrentAlgo, "SISAL"))

		# Linear Unmixing
		lu = menubar.addMenu("Linear Unmixing")

		sunsal = QAction("SUNSAL", self)
		lu.addAction(sunsal)
		sunsal.triggered.connect(partial(self.changeCurrentAlgo, "SUNSAL"))

		vca = QAction("VCA", self)
		lu.addAction(vca)
		vca.triggered.connect(partial(self.changeCurrentAlgo, "VCA"))

		nnls = QAction("NNLS", self)
		lu.addAction(nnls)
		nnls.triggered.connect(partial(self.changeCurrentAlgo, "NNLS"))

		ucls = QAction("UCLS", self)
		lu.addAction(ucls)
		ucls.triggered.connect(partial(self.changeCurrentAlgo, "UCLS"))

		fcls = QAction("FCLS", self)
		lu.addAction(fcls)
		fcls.triggered.connect(partial(self.changeCurrentAlgo, "FCLS"))

		# Non-linear Unmixing
		nlu = menubar.addMenu("Non Linear Unmixing")

		gbmNMF = QAction("GBM using semi-NMF", self)
		nlu.addAction(gbmNMF)

		gbmGrad = QAction("GBM using gradient", self)
		nlu.addAction(gbmGrad)

		mulLin = QAction("Multi-Linear", self)
		nlu.addAction(mulLin)

		kerBase = QAction("Kernel Base", self)
		nlu.addAction(kerBase)

		graphLapBase = QAction("Graph Laplacian Base", self)
		nlu.addAction(graphLapBase)


	def changeCurrentAlgo(self, algo):
		'''
		Changes the value of variable currentAlgo with the algorithm selected
		from the menu bar
		'''

		self.currentAlgo = algo


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

		# Starting selected algorithm if everything's good
		if self.dataExists and self.outputFolderExists and self.trueComponents and self.enoughProcs:

			if self.currentAlgo == "PCA":
				self.logs.addItem(f'Starting Principal Component Analysis for getting top {self.components.toPlainText()} bands')
				self.startPCA(selectedComponents)

			elif self.currentAlgo == "NMF":
				self.logs.addItem(f'Starting NMF for getting top {self.components.toPlainText()} bands')
				newpid1 = os.fork()
				if newpid1 == 0:
					nmf_data = self.startNMF(selectedComponents)

			elif self.currentAlgo == "NFinder":
				self.logs.addItem(f'Starting N-Finder for getting top {self.components.toPlainText()} bands')
				self.startHfcVd(selectedComponents)
				self.startNFINDR(self.pca_data)
				# self.startSUNSAL(self.nfindr_data, self.Et)

			elif self.currentAlgo == "SUNSAL":
				self.logs.addItem(f'Starting SUNSAL for getting estimated abundance matrix')
				self.startHfcVd(selectedComponents)
				self.startNFINDR(self.pca_data)
				self.startSUNSAL(self.nfindr_data)

			elif self.currentAlgo == "HfcVd":
				self.logs.addItem(f'Starting HfcVd for getting number of end members')
				self.startHfcVd(selectedComponents)

			elif self.currentAlgo == "ATGP":
				self.logs.addItem(f'Starting ATGP for getting Endmember abundances')
				self.startHfcVd(selectedComponents)
				self.startATGP(self.pca_data)

			elif self.currentAlgo == "VCA":
				self.logs.addItem(f'Starting VCA for getting estimated endmembers signature matrix')
				self.startHfcVd(selectedComponents)
				self.startNFINDR(self.pca_data)
				self.startVCA(self.nfindr_data)

			elif self.currentAlgo == "NNLS":
				self.logs.addItem(f'Starting NNLS for getting estimated abundance matrix')
				self.startHfcVd(selectedComponents)
				self.startNFINDR(self.pca_data)
				self.startNNLS(self.pca_data, self.nfindr_data)

			elif self.currentAlgo == "UCLS":
				self.logs.addItem(f'Starting UCLS for getting estimated abundance matrix')
				self.startHfcVd(selectedComponents)
				self.startNFINDR(self.pca_data)
				self.startUCLS(self.pca_data, self.nfindr_data)

			elif self.currentAlgo == "FCLS":
				self.logs.addItem(f'Starting FCLS for getting estimated abundance matrix')
				self.startHfcVd(selectedComponents)
				self.startNFINDR(self.pca_data)
				self.startFCLS(self.pca_data, self.nfindr_data)

			elif self.currentAlgo == "KerPCA":
				self.logs.addItem(f'Starting Kernel Principal Component Analysis for getting top {self.components.toPlainText()} bands')
				self.startKerPCA(selectedComponents)

			elif self.currentAlgo == "LLE":
				self.logs.addItem(f'Starting Locally Linear Embedding algorithm for getting top {self.components.toPlainText()} bands')
				self.startLLE(selectedComponents)

			elif self.currentAlgo == "PPI":
				self.logs.addItem(f'Starting Pixel Purity-based Algorithm for endmember extraction')
				self.startHfcVd(selectedComponents)
				self.startPPI(self.pca_data)

			elif self.currentAlgo == "Hysime":
				self.logs.addItem(f'Starting Hysime Algorithm for endmember extraction')
				self.startPCA(selectedComponents)
				self.startHysime(self.pca_data)


		self.progress.setRange(0,1)
		

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

		# t1 = Thread(target=PCAThread2)
		self.datasetAsArray = self.dataset.ReadAsArray()
		pca = PrincipalComponentAnalysis(self.datasetAsArray)
		pca.scaleData()

		# t1.start()
		self.pca_data = pca.getPrincipalComponents_noOfComponents((int)(self.components.toPlainText()))
		retainedVariance = pca.getRetainedVariance((int)(self.components.toPlainText()))
		
		self.logs.addItem("Analysis completed")
		self.logs.addItem("Generating Output file")
		self.writeData("PCA_", self.pca_data)
		# t1.join()
		self.logs.addItem(f"Output file PCA_{self.OUTPUT_FILENAME} generated")
		self.logs.addItem(f'Retained Variance: {retainedVariance}')
		self.setProgressBar(False)
		
		''' To plot the points after PCA '''
		if (int)(selectedComponents) == 1:
			newpid = os.fork()
			if newpid == 0:
				self.plot1DGraph(self.pca_data)

		elif (int)(selectedComponents) == 2:
			newpid = os.fork()
			if newpid == 0:
				self.plot2DGraph(self.pca_data)

		elif (int)(selectedComponents) == 3:
			newpid = os.fork()
			if newpid == 0:
				self.plot3DGraph(self.pca_data)

		else:
			self.logs.addItem('Due to high dimentionality, graph could not be plotted')

	# def PCAThread2():
	# 	subprocess.call()
	# 	self.pca_data = pca.getPrincipalComponents_noOfComponents((int)(self.components.toPlainText()))
	# 	self.retainedVariance = pca.getRetainedVariance((int)(self.components.toPlainText()))
	# 	self.writeData("PCA_", self.pca_data)


	def startKerPCA(self, selectedComponents):
		'''
		Main function for Kernel PCA
		'''

		self.datasetAsArray = self.dataset.ReadAsArray()
		kernelpca = KernelPCAAlgorithm(self.datasetAsArray, (int)(self.jobs.toPlainText()))
		kernelpca.scaleData()

		self.ker_pca_data = kernelpca.getPrincipalComponents_noOfComponents((int)(self.components.toPlainText()))
		
		self.logs.addItem("Analysis completed")
		self.logs.addItem("Generating Output file")
		self.writeData("KernelPCA_", self.ker_pca_data)
		self.logs.addItem(f"Output file KernelPCA_{self.OUTPUT_FILENAME} generated")
		self.setProgressBar(False)
		
		''' To plot the points after Kernel PCA '''
		if (int)(selectedComponents) == 1:
			newpid = os.fork()
			if newpid == 0:
				self.plot1DGraph(self.ker_pca_data)

		elif (int)(selectedComponents) == 2:
			newpid = os.fork()
			if newpid == 0:
				self.plot2DGraph(self.ker_pca_data)

		elif (int)(selectedComponents) == 3:
			newpid = os.fork()
			if newpid == 0:
				self.plot3DGraph(self.ker_pca_data)

		else:
			self.logs.addItem('Due to high dimentionality, graph could not be plotted')


	def startLLE(self, selectedComponents):
		'''
		Main function for LLE
		'''

		self.datasetAsArray = self.dataset.ReadAsArray()
		lleAlgo = LLE(self.datasetAsArray, (int)(self.jobs.toPlainText()))
		lleAlgo.scaleData()

		self.lle_data = lleAlgo.getPrincipalComponents_noOfComponents((int)(self.components.toPlainText()))
		
		self.logs.addItem("Analysis completed")
		self.logs.addItem("Generating Output file")
		self.writeData("LLE_", self.lle_data)
		self.logs.addItem(f"Output file LLE_{self.OUTPUT_FILENAME} generated")
		self.setProgressBar(False)
		
		''' To plot the points after LDA '''
		if (int)(selectedComponents) == 1:
			newpid = os.fork()
			if newpid == 0:
				self.plot1DGraph(self.lle_data)

		elif (int)(selectedComponents) == 2:
			newpid = os.fork()
			if newpid == 0:
				self.plot2DGraph(self.lle_data)

		elif (int)(selectedComponents) == 3:
			newpid = os.fork()
			if newpid == 0:
				self.plot3DGraph(self.lle_data)

		else:
			self.logs.addItem('Due to high dimentionality, graph could not be plotted')



	def startNMF(self,selectedComponents):
		'''
		Main function for NMF
		'''

		self.datasetAsArray = self.dataset.ReadAsArray()
		nmf = NonNegativeMatrixFactorisation(self.datasetAsArray)
		nmf.scaleData()
		self.nmf_data = nmf.getReducedComponents_noOfComponents((int)(self.components.toPlainText()))
		error = nmf.errorFactor((int)(self.components.toPlainText()))
		self.logs.addItem("Analysis completed")
		self.logs.addItem(f'RMS Error: {error}')
		self.logs.addItem("Generating Output file")
		self.nmf_data = nmf.denormalizeData(self.nmf_data)
		self.writeData("NMF_", self.nmf_data)
		self.logs.addItem(f"Output file NMF_{self.OUTPUT_FILENAME} generated")
		self.setProgressBar(False)
		
		''' To plot the points after NMF '''
		if (int)(selectedComponents) == 1:
			newpid = os.fork()
			if newpid == 0:
				self.plot1DGraph(self.nmf_data)

		elif (int)(selectedComponents) == 2:
			newpid = os.fork()
			if newpid == 0:
				self.plot2DGraph(self.nmf_data)

		elif (int)(selectedComponents) == 3:
			newpid = os.fork()
			if newpid == 0:
				self.plot3DGraph(self.nmf_data)

		else:
			self.logs.addItem('Due to high dimentionality, graph could not be plotted')


	def startATGP(self, pca_data):
		'''
		Main function to run ATGP
		'''
		self.logs.addItem("Initiating ATGP algorithm")
		self.ATGP_data, IDX = eea.ATGP(np.transpose(pca_data), self.end_member_list[2])
		self.logs.addItem("Analysis completed")
		self.logs.addItem("Generating output file")
		self.writeData("ATGP_", self.ATGP_data)
		self.logs.addItem(f"Output File ATGP_{self.OUTPUT_FILENAME} generated")
		self.setProgressBar(False)


	def startNFINDR(self, pca_data):
		'''
		Main function for N-Finder algorithm
		'''

		self.datasetAsArray = self.dataset.ReadAsArray()
		nfindr = NFindrModule()
		self.nfindr_data, Et, IDX, n_iterations = nfindr.NFINDR(pca_data, self.end_member_list[2])
		self.logs.addItem("Analysis completed")
		self.logs.addItem(f'Number of iterations: {n_iterations}')
		self.logs.addItem("Generating Output file")
		self.writeData("NFinder_", self.nfindr_data)
		self.logs.addItem(f"Output file NFinder_{self.OUTPUT_FILENAME} generated")
		self.setProgressBar(False)


	def startSUNSAL(self, nfindr_data):
		'''
		Main function for SUNSAL algorithm
		'''

		ss = SUNSALModule()
		self.logs.addItem("Initiating SUNSAL algorithm")
		self.sunsal_data, res_p, res_d, sunsal_i = ss.SUNSAL(np.transpose(nfindr_data), np.transpose(self.pca_data))
		self.logs.addItem("Running SUNSAL algorithm")
		self.writeData("SUNSAL_", self.sunsal_data)
		self.logs.addItem(f"Output file SUNSAL_{self.OUTPUT_FILENAME} generated")
		self.logs.addItem(f"Number of iterations are {sunsal_i}")
		self.setProgressBar(False)


	def startHfcVd(self, selectedComponents):
		'''
		Main function for HfcVd algorithm
		'''

		self.startPCA(selectedComponents)
		self.logs.addItem("Initiating HfcVd algorithm")
		self.end_member_list = vd.HfcVd(self.pca_data)
		self.logs.addItem("Running SUNSAL algorithm")
		self.logs.addItem(f"Number of end member(s) found is/are {self.end_member_list[2]}")
		self.setProgressBar(False)


	def startVCA(self, nfindr_data):
		'''
		Main function for VCA algorithm
		'''

		self.logs.addItem("Initiating VCA algorithm")
		self.vca_data, IDX, proj_data = sparse.vca(nfindr_data, self.end_member_list[2])
		self.logs.addItem("Running VCA algorithm")
		self.writeData("VCA_", self.vca_data)
		self.logs.addItem(f"Output file VCA_{self.OUTPUT_FILENAME} generated")
		self.setProgressBar(False)


	def startPPI(self, pca_data):
		'''
		Main function for PPI algorithm
		'''

		self.logs.addItem("Initiating PPI algorithm")
		self.ppi_data, IDX = eea.PPI(np.transpose(pca_data), self.end_member_list[2])
		self.logs.addItem("Running PPI algorithm")
		self.writeData("PPI_", self.ppi_data)
		self.logs.addItem(f"Output file PPI_{self.OUTPUT_FILENAME} generated")
		self.setProgressBar(False)


	def startHysime(self, pca_data):
		'''
		Main function for PPI algorithm
		'''

		self.logs.addItem("Initiating Hysime algorithm")
		HYSIME = Hysime()
		self.hysime_data, Ek = HYSIME.count(np.transpose(pca_data))
		self.logs.addItem("Running Hysime algorithm")
		self.writeData("Hysime_", self.hysime_data)
		self.logs.addItem(f"Output file Hysime_{self.OUTPUT_FILENAME} generated")
		self.setProgressBar(False)


	def startNNLS(self, pca_data, nfindr_data):
		'''
		Main function for NNLS algorithm
		'''

		self.logs.addItem("Initiating NNLS algorithm")
		self.NNLS_data = LMM.NNLS(pca_data, nfindr_data)
		self.logs.addItem("Analysis completed")
		self.logs.addItem("Generating output file")
		self.writeData("NNLS_", self.NNLS_data)
		self.logs.addItem(f"Output File NNLS_{self.OUTPUT_FILENAME} generated")
		self.setProgressBar(False)


	def startUCLS(self, pca_data, nfindr_data):
		'''
		Main function for UCLS algorithm
		'''

		self.logs.addItem("Initiating UCLS algorithm")
		self.UCLS_data = LMM.UCLS(pca_data, nfindr_data)
		self.logs.addItem("Analysis completed")
		self.logs.addItem("Generating output file")
		self.writeData("UCLS_", self.UCLS_data)
		self.logs.addItem(f"Output File UCLS_{self.OUTPUT_FILENAME} generated")
		self.setProgressBar(False)


	def startFCLS(self, pca_data, nfindr_data):
		'''
		Main function for FCLS algorithm
		'''

		self.logs.addItem("Initiating FCLS algorithm")
		self.UCLS_data = LMM.FCLS(pca_data, nfindr_data)
		self.logs.addItem("Analysis completed")
		self.logs.addItem("Generating output file")
		self.writeData("FCLS_", self.UCLS_data)
		self.logs.addItem(f"Output File FCLS_{self.OUTPUT_FILENAME} generated")
		self.setProgressBar(False)


	def writeData(self, prefix, data):
		'''
		Writes data into a file in CSV (Comma Seperated Value) format
		'''

		with open(prefix + self.OUTPUT_FILENAME, 'w') as writeFile:
			writer = csv.writer(writeFile)

			dataList = []
			for row in data:
				temp = []
				for cols in row:
					temp.append(cols)
				dataList.append(temp)
			writer.writerows(dataList)

		writeFile.close()


	def plot1DGraph(self, data):
		'''
		Plots one dimensional data
		'''

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


	def plot2DGraph(self, data):
		'''
		Plots two dimensional data
		'''

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


	def plot3DGraph(self, data):
		'''
		Plots three dimensional data
		'''

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

class DimensionalityReduction(Qt):



if __name__ == "__main__":

	app = QApplication(sys.argv)
	window = Software()
	window.show()

	# Adding error handler
	#std_err_handler = StdErrHandler()
	#sys.stderr = std_err_handler
	#std_err_handler.err_msg.connect(window.writeError)
sys.exit(app.exec_())