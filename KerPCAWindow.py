import sys, os

from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QPushButton, QTextEdit, QListWidget, QProgressBar, QLabel, QAction, QComboBox, QCheckBox
from PyQt5 import QtGui

path = os.path.dirname(__file__)
qtCreatorFile = "KerPCA.ui" 

Ui_MainWindow, QtBaseClass = uic.loadUiType(path + qtCreatorFile)


class Software(QMainWindow, Ui_MainWindow):

	def __init__(self):
		
		super(Software, self).__init__()
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		self.title = "Ker PCA"
		
		self.initUI()

	def initUI(self):

		self.setWindowTitle(self.title)

		# Input Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 30)

		# Input browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,30)

		# Input text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,35,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 91)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,91)

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,95,402,21)
		self.output_text.setText(os.getcwd())

		# No of components Label
		self.components_label = QLabel("Components", self)
		self.components_label.move(20, 145)

		# Components text field
		self.components = QTextEdit(self)
		self.components.setGeometry(110,150,40,21)

		# Jobs Label
		self.jobs_label = QLabel("No of jobs", self)
		self.jobs_label.move(170, 145)

		# Jobs text field
		self.jobs = QTextEdit(self)
		self.jobs.setGeometry(250,150,40,21)

		# Kernel Label
		self.kernel_label = QLabel("Kernel", self)
		self.kernel_label.move(315, 145)

		# Kernel Choice List
		self.kernelChoiceList = QComboBox(self)
		self.kernelChoiceList.addItem("linear")
		self.kernelChoiceList.addItem("poly")
		self.kernelChoiceList.addItem("rbf")
		self.kernelChoiceList.addItem("sigmoid")
		self.kernelChoiceList.addItem("cosine")
		self.kernelChoiceList.addItem("precomputed")
		self.kernelChoiceList.move(360, 145)

		# Fit inverse transform Label
		self.eigen_solver = QLabel("Eigen Solv", self)
		self.eigen_solver.move(480, 145)

		# Fit inverse transform Choice List
		self.solverChoiceList = QComboBox(self)
		self.solverChoiceList.addItem("auto")
		self.solverChoiceList.addItem("dense")
		self.solverChoiceList.addItem("arpack")
		self.solverChoiceList.move(550, 145)

		# Alpha Label
		self.alpha_label = QLabel("Alpha", self)
		self.alpha_label.move(20, 200)

		# Alpha text field
		self.alpha = QTextEdit(self)
		self.alpha.setGeometry(70,205,40,21)

		# Gamma Label
		self.gamma_label = QLabel("Gamma", self)
		self.gamma_label.move(130, 200)

		# Gamma text field
		self.gamma = QTextEdit(self)
		self.gamma.setGeometry(190,205,40,21)

		# Fit Inverse Transform Label
		self.fit_inverse_transform = QLabel("Fit Inverse Transform", self)
		self.fit_inverse_transform.setGeometry(260,210,280,15)
		
		# Fit Inverse Transform Checkbox
		self.checkbox_fit_inverse_transform = QCheckBox(self)
		self.checkbox_fit_inverse_transform.move(400, 203)

		# Remove Zero Eigen Label
		self.remove_zero_eigen = QLabel("Remove Zero Eigen", self)
		self.remove_zero_eigen.setGeometry(440,210,280,15)

		# Remove Zero Eigen Checkbox
		self.checkbox_remove_zero_eigen = QCheckBox(self)
		self.checkbox_remove_zero_eigen.move(570, 203)

		# OK button
		self.OK = QPushButton("OK", self)
		self.OK.move(230, 250)

		# Cancel button
		self.cancel = QPushButton("Cancel", self)
		self.cancel.move(380, 250)

		# Logs entry
		self.logs = QListWidget(self)
		self.logs.setGeometry(10, 300, 640, 130)

		# Progress Bar
		self.progress = QProgressBar(self)
		self.progress.setGeometry(10, 450, 640, 20)

		self.show()


if __name__ == "__main__":

	app = QApplication(sys.argv)
	window = Software()
	window.show()

sys.exit(app.exec_())