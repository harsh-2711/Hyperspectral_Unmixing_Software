import sys
import os

from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QPushButton, QTextEdit, QListWidget, QProgressBar, QLabel

from osgeo import gdal

from PCA import PrincipalComponentAnalysis
from Threads import ValidationThread


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
        self.OUTPUT_FILENAME = "PCA_Data"

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
        if fileName:
            self.input_text.setText(fileName)

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

    	# Validating dataset path
    	self.dataExists = self.validateInputFile(filename)

    	# Validating output folder path
    	if self.dataExists:
    		self.outputFolderExists = self.validateOutputFolder(foldername)

        # Validating number of components
    	if self.dataExists and self.outputFolderExists:
    		self.trueComponents = self.validateComponents(selectedComponents)

        # Start PCA if everything's good
    	if self.dataExists and self.outputFolderExists and self.trueComponents:
    		self.logs.addItem(f'Starting Principal Component Analysis for getting top {self.components.toPlainText()} bands')
    		self.startPCA()
        
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

    def startPCA(self):
    	'''
		Main function for PCA
    	'''

    	self.datasetAsArray = self.dataset.ReadAsArray()
    	pca = PrincipalComponentAnalysis(self.datasetAsArray)
    	pca_data = pca.getPrincipalComponents_noOfComponents((int)(self.components.toPlainText()))
    	self.logs.addItem("Analysis completed")
    	self.logs.addItem("Generating Output file")
    	pca_data.tofile(self.OUTPUT_FILENAME, ",")
    	self.logs.addItem("Output file generated")
    	self.setProgressBar(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Software()
    window.show()
    sys.exit(app.exec_())