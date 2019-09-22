
import sys
import os

from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot

from functools import partial

from mpl_toolkits.mplot3d import Axes3D

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QPushButton, QTextEdit, QListWidget, QProgressBar, QLabel, QAction, QWidget, QComboBox, QCheckBox
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

from Modules.End_Member_Extraction import eea
from Modules.Linear_Unmixing import sparse, LMM
from Modules.Non_Linear_Unmixing import GBM_semiNMF, GBM_GDA

from Threads import ValidationThread
from threading import Thread
import subprocess
from ErrorHandler import StdErrHandler

import multiprocessing as mp
import matplotlib.pyplot as plt


path = os.path.dirname(__file__)
qtCreatorFile = "MainWindow.ui" 

Ui_MainWindow, QtBaseClass = uic.loadUiType(path + qtCreatorFile)


currentAlgo = ""
OUTPUT_FILENAME = "Data.csv"

end_member_list = [0,0,0]


class PCAUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(PCAUI, self).__init__()

		# HSI Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 60)

		# HSI browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,60)
		self.input_browse.clicked.connect(partial(on_click_input, self))

		# HSI text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,65,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 120)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,120)
		self.output_browse.clicked.connect(partial(on_click_output, self))

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,125,402,21)
		self.output_text.setText(os.getcwd())

		# Tolerance Label		
		self.components_label = QLabel("No of Components", self)
		self.components_label.move(50, 180)

		# Components text field
		self.components = QTextEdit(self)
		self.components.setGeometry(180,185,40,21)

		# OK button
		self.getVar = QPushButton("Get Variance", self)
		self.getVar.move(280, 180)
		self.getVar.clicked.connect(partial(on_click_getVar, self))

		# OK button
		self.OK = QPushButton("OK", self)
		self.OK.move(230, 250)
		self.OK.clicked.connect(partial(on_click_OK, self))

		# Cancel button
		self.cancel = QPushButton("Cancel", self)
		self.cancel.move(380, 250)
		self.cancel.clicked.connect(partial(on_click_cancel, self))

		# Logs entry
		self.logs = QListWidget(self)
		self.logs.setGeometry(10, 300, 640, 130)

		self.varDisplay = QListWidget(self)
		self.varDisplay.setGeometry(410, 185, 180, 20)

		# Progress Bar
		self.progress = QProgressBar(self)
		self.progress.setGeometry(10, 450, 640, 20)
		
		self.show()


class KerPCAUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(KerPCAUI, self).__init__()

		# Input Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 30)

		# Input browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,30)
		self.input_browse.clicked.connect(partial(on_click_input, self))

		# Input text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,35,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 91)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,91)
		self.output_browse.clicked.connect(partial(on_click_output, self))

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
		self.kernel = self.kernelChoiceList.currentText()

		# Fit inverse transform Label
		self.eigen_solver = QLabel("Eigen Solv", self)
		self.eigen_solver.move(480, 145)

		# Fit inverse transform Choice List
		self.solverChoiceList = QComboBox(self)
		self.solverChoiceList.addItem("auto")
		self.solverChoiceList.addItem("dense")
		self.solverChoiceList.addItem("arpack")
		self.solverChoiceList.move(550, 145)
		self.solver = self.solverChoiceList.currentText()

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
		self.OK.clicked.connect(partial(on_click_OK, self))

		# Cancel button
		self.cancel = QPushButton("Cancel", self)
		self.cancel.move(380, 250)
		self.cancel.clicked.connect(partial(on_click_cancel, self))

		# Logs entry
		self.logs = QListWidget(self)
		self.logs.setGeometry(10, 300, 640, 130)

		# Progress Bar
		self.progress = QProgressBar(self)
		self.progress.setGeometry(10, 450, 640, 20)

		self.show()

class NFINDRUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(NFINDRUI, self).__init__()


		# Input Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 30)

		# Input browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,30)
		self.input_browse.clicked.connect(partial(on_click_input, self))

		# Input text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,35,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 100)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,100)
		self.output_browse.clicked.connect(partial(on_click_output, self))

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,105,402,21)
		self.output_text.setText(os.getcwd())

		# No of endmembers Label
		self.components_label = QLabel("Components", self)
		self.components_label.move(20, 200)

		# Endmembers text field
		self.components = QTextEdit(self)
		self.components.setGeometry(140,205,45,21)

		# Max iterations Label
		self.maxit_label = QLabel("Max iterations", self)
		self.maxit_label.move(220, 200)

		# Max iterations text field
		self.maxit = QTextEdit(self)
		self.maxit.setGeometry(350,205,40,21)

		# ATGP label
		self.ATGP_label = QLabel("ATGP", self)
		self.ATGP_label.setGeometry(440,208,280,15)
		
		# ATGP Checkbox
		self.checkbox_ATGP = QCheckBox(self)
		self.checkbox_ATGP.move(500, 202)

		# OK button
		self.OK = QPushButton("OK", self)
		self.OK.move(230, 250)
		self.OK.clicked.connect(partial(on_click_OK, self))

		# Cancel button
		self.cancel = QPushButton("Cancel", self)
		self.cancel.move(380, 250)
		self.cancel.clicked.connect(partial(on_click_cancel, self))

		# Logs entry
		self.logs = QListWidget(self)
		self.logs.setGeometry(10, 300, 640, 130)

		# Progress Bar
		self.progress = QProgressBar(self)
		self.progress.setGeometry(10, 450, 640, 20)

		self.show()


class LLEUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(LLEUI, self).__init__()


		# Input Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 20)

		# Input browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,20)
		self.input_browse.clicked.connect(partial(on_click_input, self))

		# Input text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,25,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 70)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,70)
		self.output_browse.clicked.connect(partial(on_click_output, self))

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,75,402,21)
		self.output_text.setText(os.getcwd())

		# No of endmembers Label
		self.components_label = QLabel("Components", self)
		self.components_label.move(20, 120)

		# Endmembers text field
		self.components = QTextEdit(self)
		self.components.setGeometry(140,125,45,21)

		# Components Label
		self.neighbours_label = QLabel("Neighbours", self)
		self.neighbours_label.move(240, 120)

		# Components text field
		self.neighbours = QTextEdit(self)
		self.neighbours.setGeometry(340,125,40,21)

		# Jobs Label
		self.jobs_label = QLabel("Jobs", self)
		self.jobs_label.move(450, 120)

		# Jobs text field
		self.jobs = QTextEdit(self)
		self.jobs.setGeometry(510,125,40,21)

		# Eigen solver Label
		self.eigenSolver_label = QLabel("Eigen solver", self)
		self.eigenSolver_label.move(140, 180)

		# Eigen Solver Choice List
		self.eigenSolverChoiceList = QComboBox(self)
		self.eigenSolverChoiceList.addItem("auto")
		self.eigenSolverChoiceList.addItem("arpack")	
		self.eigenSolverChoiceList.addItem("dense")
		self.eigenSolverChoiceList.move(230, 180)
		self.eigenSolver = self.eigenSolverChoiceList.currentText()
		
		# Method Label
		self.method_label = QLabel("Method", self)
		self.method_label.move(400, 180)

		# Method Choice List
		self.methodChoiceList = QComboBox(self)
		self.methodChoiceList.addItem("standard")
		self.methodChoiceList.addItem("hessian")
		self.methodChoiceList.addItem("modified")
		self.methodChoiceList.addItem("ltsa")
		self.methodChoiceList.move(465, 180)
		self.method = self.methodChoiceList.currentText()

		# OK button
		self.OK = QPushButton("OK", self)
		self.OK.move(230, 250)
		self.OK.clicked.connect(partial(on_click_OK, self))

		# Cancel button
		self.cancel = QPushButton("Cancel", self)
		self.cancel.move(380, 250)
		self.cancel.clicked.connect(partial(on_click_cancel, self))

		# Logs entry
		self.logs = QListWidget(self)
		self.logs.setGeometry(10, 300, 640, 130)

		# Progress Bar
		self.progress = QProgressBar(self)
		self.progress.setGeometry(10, 450, 640, 20)

		self.show()


class SUNSALUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(SUNSALUI, self).__init__()


		# Endmember sign Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 20)

		# Endmember sign browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,20)
		self.input_browse.clicked.connect(partial(on_click_input, self))

		# Endmember sign text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,25,402,21)

		# Data Matrix text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,75,402,21)
		self.output_text.setText(os.getcwd())

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 120)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,120)
		self.output_browse.clicked.connect(partial(on_click_output, self))

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,125,402,21)
		self.output_text.setText(os.getcwd())

		# No of iterations Label
		self.iter_label = QLabel("Min AL iter", self)
		self.iter_label.move(60, 170)

		# iterations text field
		self.iter = QTextEdit(self)
		self.iter.setGeometry(140,175,45,21)

		# Components Label
		self.lambda_label = QLabel("Lambda", self)
		self.lambda_label.move(260, 170)

		# Components text field
		self.lambda_val = QTextEdit(self)
		self.lambda_val.setGeometry(320,175,40,21)

		# Tolerance Label
		self.tol_label = QLabel("Tolerance", self)
		self.tol_label.move(430, 170)

		# Tolerance text field
		self.tolerance = QTextEdit(self)
		self.tolerance.setGeometry(510,175,40,21)


		# Positivity label
		self.positivity_label = QLabel("Positivity", self)
		self.positivity_label.setGeometry(130,212,280,15)
		
		# Positivity Checkbox
		self.positivity = QCheckBox(self)
		self.positivity.move(200, 206)

		# Addone label
		self.addOne_label = QLabel("Add one", self)
		self.addOne_label.setGeometry(250,212,280,15)
		
		# Addone Checkbox
		self.addOne = QCheckBox(self)
		self.addOne.move(320, 206)

		# Verbose label
		self.verbose_label = QLabel("Verbose", self)
		self.verbose_label.setGeometry(370,212,280,15)
		
		# Verbose Checkbox
		self.verbose = QCheckBox(self)
		self.verbose.move(440, 206)

		# OK button
		self.OK = QPushButton("OK", self)
		self.OK.move(230, 250)
		self.OK.clicked.connect(partial(on_click_OK, self))

		# Cancel button
		self.cancel = QPushButton("Cancel", self)
		self.cancel.move(380, 250)
		self.cancel.clicked.connect(partial(on_click_cancel, self))

		# Logs entry
		self.logs = QListWidget(self)
		self.logs.setGeometry(10, 300, 640, 130)

		# Progress Bar
		self.progress = QProgressBar(self)
		self.progress.setGeometry(10, 450, 640, 20)
		
		self.show()


class HFCVDUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(HFCVDUI, self).__init__()


		# HSI Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 60)

		# HSI text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,65,402,21)
		self.input_text.setPlaceholderText('Input PCA data here')
		self.input_text.setTabChangesFocus(True)

		# HSI browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,60)
		self.input_browse.clicked.connect(partial(on_click_input, self))
		self.input_text.setTabChangesFocus(True)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 120)

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,125,402,21)
		self.output_text.setText(os.getcwd())
		self.input_text.setTabChangesFocus(True)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,120)
		self.output_browse.clicked.connect(partial(on_click_output, self))
		self.input_text.setTabChangesFocus(True)

		# OK button
		self.OK = QPushButton("OK", self)
		self.OK.move(230, 250)
		self.OK.clicked.connect(partial(on_click_OK, self))
		self.input_text.setTabChangesFocus(True)

		# Cancel button
		self.cancel = QPushButton("Cancel", self)
		self.cancel.move(380, 250)
		self.cancel.clicked.connect(partial(on_click_cancel, self))
		self.input_text.setTabChangesFocus(True)

		# Logs entry
		self.logs = QListWidget(self)
		self.logs.setGeometry(10, 300, 640, 130)

		# Progress Bar
		self.progress = QProgressBar(self)
		self.progress.setGeometry(10, 450, 640, 20)
		
		self.show()


class NMFUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(NMFUI, self).__init__()


		# HSI Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 20)

		# HSI browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,20)
		self.input_browse.clicked.connect(partial(on_click_input, self))

		# HSI text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,25,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 70)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550, 70)
		self.output_browse.clicked.connect(partial(on_click_output, self))

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,75,402,21)
		self.output_text.setText(os.getcwd())

		# Tolerance Label	
		self.components_label = QLabel("Components", self)
		self.components_label.move(70, 120)

		# Tolerance text field
		self.components = QTextEdit(self)
		self.components.setGeometry(170,125,40,21)

		# Tolerance Label		
		self.tolerance_label = QLabel("Tolerance", self)
		self.tolerance_label.move(250, 120)

		# Tolerance text field
		self.tolerance = QTextEdit(self)
		self.tolerance.setGeometry(330,125,40,21)

		# Tolerance Label		
		self.maxit_label = QLabel("Max iter", self)
		self.maxit_label.move(410, 120)

		# Tolerance text field
		self.maxit = QTextEdit(self)
		self.maxit.setGeometry(480,125,40,21)

		# Method Label
		self.method_label = QLabel("Method", self)
		self.method_label.move(100, 180)

		# Method Choice List
		self.methodChoiceList = QComboBox(self)
		self.methodChoiceList.addItem("random")
		self.methodChoiceList.addItem("nndsvd")
		self.methodChoiceList.addItem("nndsvda")
		self.methodChoiceList.addItem("nndsvdar")
		self.methodChoiceList.addItem("custom")
		self.methodChoiceList.move(165, 180)
		self.method = self.methodChoiceList.currentText()


		# Method Label
		self.solver_label = QLabel("Solver", self)
		self.solver_label.move(300, 180)

		# Method Choice List
		self.solverChoiceList = QComboBox(self)
		self.solverChoiceList.addItem("Coordinate Descent")
		self.solverChoiceList.addItem("Multiplicative Update")
		self.solverChoiceList.move(365, 180)
		self.solver = self.solverChoiceList.currentText()

		# OK button
		self.OK = QPushButton("OK", self)
		self.OK.move(230, 250)
		self.OK.clicked.connect(partial(on_click_OK, self))

		# Cancel button
		self.cancel = QPushButton("Cancel", self)
		self.cancel.move(380, 250)
		self.cancel.clicked.connect(partial(on_click_cancel, self))

		# Logs entry
		self.logs = QListWidget(self)
		self.logs.setGeometry(10, 300, 640, 130)

		# Progress Bar
		self.progress = QProgressBar(self)
		self.progress.setGeometry(10, 450, 640, 20)
		
		self.show()


class PPIUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(PPIUI, self).__init__()


		# Endmember sign Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 20)

		# Endmember sign browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,20)
		self.input_browse.clicked.connect(partial(on_click_input, self))

		# Endmember sign text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,25,402,21)

		# Data matrix Label
		self.initskewers_label = QLabel("Init Skewers", self)
		self.initskewers_label.move(20, 70)

		# Data Matrix browse button
		self.initskewers_browse = QPushButton("Browse", self)
		self.initskewers_browse.move(550,70)

		# Data Matrix text field
		self.initskewers_text = QTextEdit(self)
		self.initskewers_text.setGeometry(142,75,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 120)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,120)
		self.output_browse.clicked.connect(partial(on_click_output, self))

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,125,402,21)
		self.output_text.setText(os.getcwd())

		# Tolerance Label		
		self.components_label = QLabel("Components", self)
		self.components_label.move(130, 180)

		# Tolerance text field
		self.components = QTextEdit(self)
		self.components.setGeometry(220,185,40,21)

		# Tolerance Label		
		self.skewers_label = QLabel("No of Skewers", self)
		self.skewers_label.move(300, 180)

		# Tolerance text field
		self.skewers = QTextEdit(self)
		self.skewers.setGeometry(400,185,40,21)

		# OK button
		self.OK = QPushButton("OK", self)
		self.OK.move(230, 250)
		self.OK.clicked.connect(partial(on_click_OK, self))

		# Cancel button
		self.cancel = QPushButton("Cancel", self)
		self.cancel.move(380, 250)
		self.cancel.clicked.connect(partial(on_click_cancel, self))

		# Logs entry
		self.logs = QListWidget(self)
		self.logs.setGeometry(10, 300, 640, 130)

		# Progress Bar
		self.progress = QProgressBar(self)
		self.progress.setGeometry(10, 450, 640, 20)
		
		self.show()

class VCAUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(VCAUI, self).__init__()


		# HSI Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 45)

		# HSI browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,45)
		self.input_browse.clicked.connect(partial(on_click_input, self))

		# HSI text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,50,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 100)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,100)
		self.output_browse.clicked.connect(partial(on_click_output, self))

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,105,402,21)
		self.output_text.setText(os.getcwd())

		self.components_label = QLabel("Components", self)
		self.components_label.move(150, 150)

		# Tolerance text field
		self.components = QTextEdit(self)
		self.components.setGeometry(260,155,40,21)

		self.SNR_label = QLabel("SNR input", self)
		self.SNR_label.move(340, 150)

		# Tolerance text field
		self.SNR = QTextEdit(self)
		self.SNR.setGeometry(420,155,40,21)

		self.verbose_label = QLabel("Verbose", self)
		self.verbose_label.setGeometry(150,212,280,15)
		
		# Verbose Checkbox
		self.verbose = QCheckBox(self)
		self.verbose.move(220, 206)

		# OK button
		self.OK = QPushButton("OK", self)
		self.OK.move(230, 250)
		self.OK.clicked.connect(partial(on_click_OK, self))

		# Cancel button
		self.cancel = QPushButton("Cancel", self)
		self.cancel.move(380, 250)
		self.cancel.clicked.connect(partial(on_click_cancel, self))

		# Logs entry
		self.logs = QListWidget(self)
		self.logs.setGeometry(10, 300, 640, 130)

		# Progress Bar
		self.progress = QProgressBar(self)
		self.progress.setGeometry(10, 450, 640, 20)
		
		self.show()

class ATGPUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(ATGPUI, self).__init__()


		# HSI Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 60)

		# HSI browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,60)
		self.input_browse.clicked.connect(partial(on_click_input, self))

		# HSI text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,65,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 120)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,120)
		self.output_browse.clicked.connect(partial(on_click_output, self))

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,125,402,21)
		self.output_text.setText(os.getcwd())

		# Tolerance Label		
		self.components_label = QLabel("Components", self)
		self.components_label.move(150, 180)

		# Tolerance text field
		self.components = QTextEdit(self)
		self.components.setGeometry(280,185,40,21)


		# OK button
		self.OK = QPushButton("OK", self)
		self.OK.move(230, 250)
		self.OK.clicked.connect(partial(on_click_OK, self))

		# Cancel button
		self.cancel = QPushButton("Cancel", self)
		self.cancel.move(380, 250)
		self.cancel.clicked.connect(partial(on_click_cancel, self))

		# Logs entry
		self.logs = QListWidget(self)
		self.logs.setGeometry(10, 300, 640, 130)

		# Progress Bar
		self.progress = QProgressBar(self)
		self.progress.setGeometry(10, 450, 640, 20)
		
		self.show()


class NNLSUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(NNLSUI, self).__init__()


		# Endmember sign Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 40)

		# Endmember sign browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,40)
		self.input_browse.clicked.connect(partial(on_click_input, self))

		# Endmember sign text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,45,402,21)


		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 100)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,100)
		self.output_browse.clicked.connect(partial(on_click_output, self))

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,105,402,21)
		self.output_text.setText(os.getcwd())

		# Tolerance Label		
		self.components_label = QLabel("No of Components", self)
		self.components_label.move(150, 180)

		# Components text field
		self.components = QTextEdit(self)
		self.components.setGeometry(280,185,40,21)

		# OK button
		self.OK = QPushButton("OK", self)
		self.OK.move(230, 250)
		self.OK.clicked.connect(partial(on_click_OK, self))

		# Cancel button
		self.cancel = QPushButton("Cancel", self)
		self.cancel.move(380, 250)
		self.cancel.clicked.connect(partial(on_click_cancel, self))

		# Logs entry
		self.logs = QListWidget(self)
		self.logs.setGeometry(10, 300, 640, 130)

		# Progress Bar
		self.progress = QProgressBar(self)
		self.progress.setGeometry(10, 450, 640, 20)
		
		self.show()


class GBMsemiNMFUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(GBMsemiNMFUI, self).__init__()


		# Endmember sign Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 20)

		# Endmember sign browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,20)
		self.input_browse.clicked.connect(partial(on_click_input, self))

		# Endmember sign text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,25,402,21)

		# Data matrix Label
		self.output_label = QLabel("Endmember Mat", self)
		self.output_label.move(20, 75)

		# Data Matrix browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,75)

		# Data Matrix text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,80,402,21)
		self.output_text.setText(os.getcwd())

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 130)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,130)
		self.output_browse.clicked.connect(partial(on_click_output, self))

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,135,402,21)
		self.output_text.setText(os.getcwd())

		self.jobs_label = QLabel("Tolerance", self)
		self.jobs_label.move(150, 170)

		# Tolerance text field
		self.jobs = QTextEdit(self)
		self.jobs.setGeometry(230,175,40,21)

		# Tolerance Label		
		self.jobs_label = QLabel("Max iter", self)
		self.jobs_label.move(340, 170)

		# Tolerance text field
		self.jobs = QTextEdit(self)
		self.jobs.setGeometry(420,175,40,21)


		self.fit_inverse_transform = QLabel("Verbose", self)
		self.fit_inverse_transform.setGeometry(150,212,280,15)
		
		# Verbose Checkbox
		self.checkbox_fit_inverse_transform = QCheckBox(self)
		self.checkbox_fit_inverse_transform.move(220, 206)

		# OK button
		self.OK = QPushButton("OK", self)
		self.OK.move(230, 250)
		self.OK.clicked.connect(partial(on_click_OK, self))

		# Cancel button
		self.cancel = QPushButton("Cancel", self)
		self.cancel.move(380, 250)
		self.cancel.clicked.connect(partial(on_click_cancel, self))

		# Logs entry
		self.logs = QListWidget(self)
		self.logs.setGeometry(10, 300, 640, 130)

		# Progress Bar
		self.progress = QProgressBar(self)
		self.progress.setGeometry(10, 450, 640, 20)
		
		self.show()


class Software (QMainWindow, Ui_MainWindow):

	def __init__(self):
		'''
		Initializing software
		'''

		super(Software, self).__init__()
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		self.title = "Hyperspectral Unmixing Toolbox"
		# self.OUTPUT_FILENAME = "Data.csv"
		self.file = ""
		# self.currentAlgo = "PCA"

		# Initialising Menu Bar
		self.initMenu()

		# Initialising default UI
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
		pcaMenu.triggered.connect(partial(clearWidgets, self))
		pcaMenu.triggered.connect(partial(startPCAWindow, self))
		pcaMenu.triggered.connect(partial(self.changeCurrentAlgo, "PCA"))


		nmf = QAction("NMF", self)
		dimReduction.addAction(nmf)
		nmf.triggered.connect(partial(clearWidgets, self))
		nmf.triggered.connect(partial(startNMFWindow, self))
		nmf.triggered.connect(partial(self.changeCurrentAlgo, "NMF"))

		kerPCA = QAction("Kernel PCA", self)
		dimReduction.addAction(kerPCA)
		kerPCA.triggered.connect(partial(clearWidgets, self))
		kerPCA.triggered.connect(partial(startKerPCAWindow, self))
		kerPCA.triggered.connect(partial(self.changeCurrentAlgo, "KerPCA"))

		lda = QAction("FDA", self)
		dimReduction.addAction(lda) 

		lle = QAction("LLE", self)
		dimReduction.addAction(lle)
		lle.triggered.connect(partial(clearWidgets, self))
		lle.triggered.connect(partial(startLLEWindow, self))
		lle.triggered.connect(partial(self.changeCurrentAlgo, "LLE"))

		# Material Count
		mc = menubar.addMenu("Material Count")

		virDim = QAction("Virtual Dimension", self)
		mc.addAction(virDim)

		hysime = QAction("Hysime", self)
		mc.addAction(hysime)

		hfcvd = QAction("HfcVd", self)
		mc.addAction(hfcvd)
		hfcvd.triggered.connect(partial(clearWidgets, self))
		hfcvd.triggered.connect(partial(startHFCVDWindow, self))
		hfcvd.triggered.connect(partial(self.changeCurrentAlgo, "HfcVd"))

		# End Member Extraction
		eme = menubar.addMenu("End Member Extraction")

		nFinder = QAction("N-Finder", self)
		eme.addAction(nFinder)
		nFinder.triggered.connect(partial(clearWidgets, self))
		nFinder.triggered.connect(partial(startNFINDRWindow, self))
		nFinder.triggered.connect(partial(self.changeCurrentAlgo, "NFinder"))

		atgp = QAction("ATGP", self)
		eme.addAction(atgp)
		atgp.triggered.connect(partial(clearWidgets, self))
		atgp.triggered.connect(partial(startATGPWindow, self))
		atgp.triggered.connect(partial(self.changeCurrentAlgo, "ATGP"))

		ppi = QAction("PPI", self)
		eme.addAction(ppi)
		ppi.triggered.connect(partial(clearWidgets, self))
		ppi.triggered.connect(partial(startPPIWindow, self))
		ppi.triggered.connect(partial(self.changeCurrentAlgo, "PPI"))

		sisal = QAction("SISAL", self)
		eme.addAction(sisal)
		# sisal.triggered.connect(partial(changeCurrentAlgo, self, "SISAL"))

		# Linear Unmixing
		lu = menubar.addMenu("Linear Unmixing")

		sunsal = QAction("SUNSAL", self)
		lu.addAction(sunsal)
		sunsal.triggered.connect(partial(clearWidgets, self))
		sunsal.triggered.connect(partial(startSUNSALWindow, self))
		sunsal.triggered.connect(partial(self.changeCurrentAlgo, "SUNSAL"))

		vca = QAction("VCA", self)
		lu.addAction(vca)
		vca.triggered.connect(partial(clearWidgets, self))
		vca.triggered.connect(partial(startVCAWindow, self))
		vca.triggered.connect(partial(self.changeCurrentAlgo, "VCA"))

		nnls = QAction("NNLS", self)
		lu.addAction(nnls)
		nnls.triggered.connect(partial(clearWidgets, self))
		nnls.triggered.connect(partial(startNNLSWindow, self))
		nnls.triggered.connect(partial(self.changeCurrentAlgo, "NNLS"))

		ucls = QAction("UCLS", self)
		lu.addAction(ucls)
		ucls.triggered.connect(partial(clearWidgets, self))
		ucls.triggered.connect(partial(startUCLSWindow, self))
		ucls.triggered.connect(partial(self.changeCurrentAlgo, "UCLS"))

		fcls = QAction("FCLS", self)
		lu.addAction(fcls)
		fcls.triggered.connect(partial(clearWidgets, self))
		fcls.triggered.connect(partial(startFCLSWindow, self))
		fcls.triggered.connect(partial(self.changeCurrentAlgo, "FCLS"))

		# Non-linear Unmixing
		nlu = menubar.addMenu("Non Linear Unmixing")

		gbmNMF = QAction("GBM using semi-NMF", self)
		nlu.addAction(gbmNMF)
		gbmNMF.triggered.connect(partial(clearWidgets, self))
		gbmNMF.triggered.connect(partial(startGBMsemiNMFWindow, self))
		gbmNMF.triggered.connect(partial(self.changeCurrentAlgo, "GBM using semi-NMF"))

		gbmGrad = QAction("GBM using gradient", self)
		nlu.addAction(gbmGrad)
		gbmGrad.triggered.connect(partial(clearWidgets, self))
		gbmGrad.triggered.connect(partial(startGBMGDAWindow, self))
		gbmGrad.triggered.connect(partial(self.changeCurrentAlgo, "GBM using gradient"))

		mulLin = QAction("Multi-Linear", self)
		nlu.addAction(mulLin)

		kerBase = QAction("Kernel Base", self)
		nlu.addAction(kerBase)

		graphLapBase = QAction("Graph Laplacian Base", self)
		nlu.addAction(graphLapBase)


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
		self.input_browse.clicked.connect(partial(on_click_input, self))

		# Input text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,35,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 91)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,91)
		self.output_browse.clicked.connect(partial(on_click_output, self))

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
		self.OK.clicked.connect(partial(on_click_OK, self))

		# Cancel button
		self.cancel = QPushButton("Cancel", self)
		self.cancel.move(380, 250)
		self.cancel.clicked.connect(partial(on_click_cancel, self))

		# Logs entry
		self.logs = QListWidget(self)
		self.logs.setGeometry(10, 300, 640, 130)

		# Progress Bar
		self.progress = QProgressBar(self)
		self.progress.setGeometry(10, 450, 640, 20)

		self.show()


	def changeCurrentAlgo(self, algo):
		'''
		Changes the value of variable currentAlgo with the algorithm selected
		from the menu bar
		'''
		global currentAlgo
		currentAlgo = algo



def setProgressBar(context,switch):
	'''
	switch - Boolean 
	Switches the progress bar from busy to stop and vice versa based on the
	value of switch
	'''

	if switch:
		context.progress.setRange(0,0)
	else:
		context.progress.setRange(0,1)



def validate(context):
	'''
	Parent function for validating all the input fields
	'''

	# Suppressing printing of errors using GDAL lib
	gdal.UseExceptions()
	gdal.PushErrorHandler('CPLQuietErrorHandler')

	foldername = context.output_text.toPlainText()
	# context.outputFolderExists = validateOutputFolder(context, foldername)

	# if currentAlgo == "HfcVd":
	# 	filename = context.input_text.toPlainText()
	# 	context.dataExists = validateCSVFile(context, context.file)

	# 	if context.dataExists and context.outputFolderExists:
	# 		context.logs.addItem(f'Starting HFCVD for finding number of endmembers')
	# 		startHfcVd(context)

	# else:
	# # selectedComponents = self.components.toPlainText()
	# # n_jobs = self.jobs.toPlainText()
	# # tolerance = self.tolerance.toPlainText()
	# # max_iterations = self.maxit.toPlainText()

	# Validating dataset path
	context.dataExists = validateInputFile(context, context.file)
	# context.logs.addItem(f'Current algo = {currentAlgo}')
	selectedComponents = context.components.toPlainText()
	# Validating output folder path
	if context.dataExists:
		context.outputFolderExists = validateOutputFolder(context, foldername)

	if context.outputFolderExists and context.dataExists:
		context.trueComponents = validateComponents(context, selectedComponents)

	if context.dataExists and context.outputFolderExists and context.trueComponents:
		if currentAlgo == "PCA":
			context.logs.addItem(f'Starting Principal Component Analysis for getting top {context.components.toPlainText()} bands')
			startPCA(context, selectedComponents)

		elif currentAlgo == "KerPCA":
			context.logs.addItem(f'Starting Kernel Principal Component Analysis for dimentionality reduction')
			startKerPCA(context, selectedComponents, context.jobs, context.kernel, context.solver, context.alpha, context.gamma, context.checkbox_fit_inverse_transform, context.checkbox_remove_zero_eigen)

		elif currentAlgo == "NMF":
			context.logs.addItem(f'Starting Non-negative Matrix Factorization for getting top {context.components.toPlainText()} bands')
			startNMF(context, selectedComponents, context.tolerance, context.maxit, context.method, context.solver)

		elif currentAlgo == "HfcVd":
			context.logs.addItem(f'Starting HFCVD for finding number of endmembers')
			startHfcVd(context, selectedComponents)

		elif currentAlgo == "NNLS":
			startHfcVd(context, selectedComponents)
			startNFINDR(context, context.pca_data, None, False)
			startNNLS(context, context.pca_data, context.nfindr_data)

		elif currentAlgo == "UCLS":
			startHfcVd(context, selectedComponents)
			startNFINDR(context, context.pca_data, None, False)
			startUCLS(context, context.pca_data, context.nfindr_data)

		elif currentAlgo == "FCLS":
			startHfcVd(context, selectedComponents)
			startNFINDR(context, context.pca_data, None, False)
			startFCLS(context, context.pca_data, context.nfindr_data)

		elif currentAlgo == "LLE":
			startLLE(context, selectedComponents, context.neighbours, context.jobs, context.eigenSolver, context.method)
		
		elif currentAlgo == "NFinder":
			startHfcVd(context, selectedComponents)
			startNFINDR(context, context.pca_data, context.maxit, context.checkbox_ATGP)

		elif currentAlgo == "ATGP":
			startHfcVd(context, selectedComponents)
			startATGP(context, context.pca_data)

		elif currentAlgo == "SUNSAL":
			startHfcVd(context, selectedComponents)
			startNFINDR(context, context.pca_data, None, False)
			startSUNSAL(context, context.nfindr_data, context.iter, context.lambda_val, context.tolerance, context.positivity, context.addOne, context.verbose)		

		elif currentAlgo == "GBM using Gradient":
			startHfcVd(context, selectedComponents)
			startNFINDR(context, context.pca_data, None, False)
			startGBMGDA(context, context.pca_data, context.nfindr_data)

		elif currentAlgo == "VCA":
			startHfcVd(context, selectedComponents)
			startNFINDR(context, context.pca_data, None, False)
			startVCA(context, context.nfindr_data, context.SNR, context.verbose)

		elif currentAlgo == "PPI":
			startHfcVd(context, selectedComponents)
			startPPI(context, context.pca_data, context.skewers)


	# # Validating number of components
	# if self.dataExists and self.outputFolderExists:
	# 	self.trueComponents = validateComponents(self, selectedComponents)

	# # Validating number of jobs
	# if self.dataExists and self.outputFolderExists and self.trueComponents:
	# 	self.enoughProcs = validateJobs(self,n_jobs)

	# # Starting selected algorithm if everything's good
	# if self.dataExists and self.outputFolderExists and self.trueComponents and self.enoughProcs:

	# 	if self.currentAlgo == "PCA":
	# 		self.logs.addItem(f'Starting Principal Component Analysis for getting top {self.components.toPlainText()} bands')
	# 		startPCA(self, selectedComponents)

	# 	elif self.currentAlgo == "NMF":
	# 		self.logs.addItem(f'Starting NMF for getting top {self.components.toPlainText()} bands')
	# 		newpid1 = os.fork()
	# 		if newpid1 == 0:
	# 			nmf_data = startNMF(selectedComponents, tolerance, max_iterations)

	# 	elif self.currentAlgo == "ATGP":
	# 		self.logs.addItem(f'Starting ATGP for getting Endmember abundances')
	# 		startHfcVd(self, selectedComponents)
	# 		startATGP(self, self.pca_data)

	# 	elif self.currentAlgo == "PPI":
	# 		self.logs.addItem(f'Starting PPI for getting Endmember Extraction')
	# 		startHfcVd(self, selectedComponents)
	# 		startPPI(self, self.pca_data)

	# 	elif self.currentAlgo == "VCA":
	# 		self.logs.addItem(f'Starting VCA for getting estimated endmembers signature matrix')
	# 		startHfcVd(self, selectedComponents)
	# 		startNFINDR(self, self.pca_data)
	# 		startVCA(self, self.nfindr_data)

	# 	elif self.currentAlgo == "KerPCA":
	# 		self.logs.addItem(f'Starting Kernel Principal Component Analysis for getting top {self.components.toPlainText()} bands')
	# 		startKerPCA(self, selectedComponents)

	# 	elif self.currentAlgo == "LLE":
	# 		self.logs.addItem(f'Starting Locally Linear Embedding algorithm for getting top {self.components.toPlainText()} bands')
	# 		startLLE(self, selectedComponents)

	# 	elif self.currentAlgo == "GBM using semi-NMF":
	# 		self.logs.addItem(f'Starting Generalized Bilinear Model for Non-Linear Unmixing')
	# 		startHfcVd(self, selectedComponents)
	# 		startNFINDR(self, self.pca_data)
	# 		startGBMsemiNMF(self, self.pca_data, self.nfindr_data)

	context.progress.setRange(0,1)


@pyqtSlot()
def on_click_input(context):
	'''
	On click listener for input_browse button
	'''

	InputBrowse(context)


@pyqtSlot()
def on_click_output(context):
	'''
	On click listener for output_browse button
	'''

	OutputBrowse(context)


@pyqtSlot()
def on_click_OK(context):
	'''
	On click listener for OK button
	'''

	context.validation = ValidationThread()
	context.validation.startThread.connect(partial(validate, context))
	context.validation.startProgress.connect(partial(setProgressBar, context))
	context.validation.start()


@pyqtSlot()
def on_click_getVar(context):
	'''
	On click listener for OK button
	'''
	context.getVar = ValidationThread()
	context.getVar.startThread.connect(partial(getRetVariance, context))
	context.getVar.start()

@pyqtSlot()
def on_click_cancel(context):
	'''
	On click listener for Cancel button
	'''

	context.input_text.setText("")
	context.output_text.setText("")
	context.components.setText("")
	context.progress.setValue(0)
	context.logs.clear()

	if currentAlgo == "NMF":
		context.tolerance.setText("")
		context.maxit.setText("")

	if currentAlgo == "PCA":
		context.varDisplay.clear()

		
def InputBrowse(context):
	'''
	Opens Browse Files dialog box for selecting input dataset
	'''

	options = QFileDialog.Options()
	options |= QFileDialog.DontUseNativeDialog
	fileName, _ = QFileDialog.getOpenFileName(context,"Select Dataset", "","All Files (*);;Matlab Files (*.mat)", options=options)
	context.file = fileName
	if fileName:
		context.input_text.setText(fileName.split('/')[-1])


def getRetVariance(context):

	# Suppressing printing of errors using GDAL lib
	gdal.UseExceptions()
	gdal.PushErrorHandler('CPLQuietErrorHandler')

	filename = context.input_text.toPlainText()
	foldername = context.output_text.toPlainText()
	context.dataExists = validateInputFile(context, context.file)
	selectedComponents = context.components.toPlainText()

	context.datasetAsArray = context.dataset.ReadAsArray()
	pca = PrincipalComponentAnalysis(context.datasetAsArray)
	pca.scaleData()

	retainedVariance = pca.getRetainedVariance((int)(context.components.toPlainText()))
	
	context.varDisplay.addItem(f'{retainedVariance}')



def OutputBrowse(context):
	'''
	Opens Browse Files dialog box for selecting target file for writing output
	'''

	options = QFileDialog.Options()
	options |= QFileDialog.DontUseNativeDialog
	folderName = str(QFileDialog.getExistingDirectory(context, "Select Directory", options=options))
	if folderName:
		context.output_text.setText(folderName)

def validateInputFile(context, filename):
	'''
	Validates the dataset path and loads the dataset if path exists
	'''

	if filename:
		try:
			context.dataset = gdal.Open(filename, gdal.GA_ReadOnly)
			context.logs.addItem("Dataset imported successfully")
			return True
		except:
			context.logs.addItem(gdal.GetLastErrorMsg())
			context.logs.addItem('Use command line argument gdalinfo --formats to get more insights')
			return False
	else:
		context.logs.addItem("Please provide path to dataset")
		return False

def validateCSVFile(context, filename):

	if filename:
		try:
			ifile = open(filename, "rU")
			reader = csv.reader(ifile, delimiter=";")
			rownum = 0	
			context.dataset = []
			for row in reader:
				context.dataset.append(row)
				rownum += 1
			ifile.close()		
			context.logs.addItem("Dataset imported successfully")
			return True
			
		except:
			context.logs.addItem(gdal.GetLastErrorMsg())
			context.logs.addItem('Reading error')
			return False
	else:
		context.logs.addItem("Please provide path to dataset")
		return False


def validateOutputFolder(context, foldername):
	'''
	Validates the existence of output folder where outfile file will be 
	created after analysis
	'''

	if foldername:
		if os.path.isdir(foldername):
			return True
	context.logs.addItem("Please provide a valid directory to save output file")
	return False


def validateComponents(context, selectedComponents):
	'''
	Validates the number of components w.r.t. the input dataset
	'''

	totalComponents = context.dataset.RasterCount
	if selectedComponents.isdigit():
		if (int)(selectedComponents) > 0 and (int)(selectedComponents) <= totalComponents:
			return True
	context.logs.addItem(f'Incorrect number of bands... Max possible number of bands are {totalComponents}')
	return False


def validateJobs(context, n_jobs):
	'''
	Validates the number of jobs desired as per processors available
	'''

	n_processors = mp.cpu_count()
	if n_jobs.isdigit():
		if (int)(n_jobs) > 0 and (int)(n_jobs) <= n_processors:
			return True
	context.logs.addItem(f'Number of jobs must be greater than 0 and less than {n_processors}')
	return False


def startLLE(context, selectedComponents, neighbours, jobs, eigenSolver, method):
	'''
	Main function for LLE
	'''

	typeCastedJobs = 0
	if jobs.toPlainText() == "" or not is_number(jobs.toPlainText()):
		typeCastedJobs = -1
	else:
		typeCastedJobs = (int)(jobs.toPlainText())

	context.datasetAsArray = context.dataset.ReadAsArray()
	lleAlgo = LLE(context.datasetAsArray, (int)(context.jobs.toPlainText()))
	lleAlgo.scaleData()

	context.lle_data = lleAlgo.getPrincipalComponents_noOfComponents((int)(selectedComponents.toPlainText()), (int)(neighbours.toPlainText()), (int)(jobs.toPlainText()), eigenSolver, method)
	
	context.logs.addItem("Analysis completed")
	context.logs.addItem("Generating Output file")
	writeData(context, "LLE_", context.lle_data)
	context.logs.addItem(f"Output file LLE_{OUTPUT_FILENAME} generated")
	setProgressBar(False)
	
	''' To plot the points after LDA '''
	if (int)(selectedComponents) == 1:
		newpid = os.fork()
		if newpid == 0:
			plot1DGraph(context,context.lle_data)

	elif (int)(selectedComponents) == 2:
		newpid = os.fork()
		if newpid == 0:
			plot2DGraph(context,context.lle_data)

	elif (int)(selectedComponents) == 3:
		newpid = os.fork()
		if newpid == 0:
			plot3DGraph(context,context.lle_data)
	else:
		context.logs.addItem('Due to high dimentionality, graph could not be plotted')


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def startNMF(context,selectedComponents, tolerance, max_iterations, method, solver):
	'''
	Main function for NMF
	'''

	context.datasetAsArray = context.dataset.ReadAsArray()
	nmf = NonNegativeMatrixFactorisation(context.datasetAsArray)
	nmf.scaleData()

	typeCastedTol = 0
	if tolerance.toPlainText() == "" or not is_number(tolerance.toPlainText()):
		typeCastedTol = 1e-4
	else:
		typeCastedTol = (float)(tolerance.toPlainText())

	typeCastedIter = 0
	if max_iterations.toPlainText() == "" or not is_number(max_iterations.toPlainText()):
		typeCastedIter = 200
	else:
		typeCastedIter = (int)(max_iterations.toPlainText())
	
	context.nmf_data = nmf.getReducedComponents_noOfComponents((int)(context.components.toPlainText()), typeCastedTol, typeCastedIter, method, solver)
	
	context.logs.addItem("Analysis completed")
	context.logs.addItem("Generating Output file")
	writeData(context, "NMF_", context.nmf_data)
	context.logs.addItem(f"Output file NMF_{OUTPUT_FILENAME} generated")
	setProgressBar(context, False)

	''' To plot the points after NMF '''
	if (int)(selectedComponents) == 1:
		newpid = os.fork()
		if newpid == 0:
			plot1DGraph(context, context.nmf_data)

	elif (int)(selectedComponents) == 2:
		newpid = os.fork()
		if newpid == 0:
			plot2DGraph(context, context.nmf_data)

	elif (int)(selectedComponents) == 3:
		newpid = os.fork()
		if newpid == 0:
			plot3DGraph(context, context.nmf_data)

	else:
		context.logs.addItem('Due to high dimentionality, graph could not be plotted')


def startATGP(context, pca_data):
	'''
	Main function to run ATGP
	'''
	context.logs.addItem("Initiating ATGP algorithm")
	context.ATGP_data, IDX = eea.ATGP(np.transpose(pca_data), end_member_list[2])
	context.logs.addItem("Analysis completed")
	context.logs.addItem("Generating output file")
	writeData(context, "ATGP_", context.ATGP_data)
	context.logs.addItem(f"Output File ATGP_{OUTPUT_FILENAME} generated")
	context.setProgressBar(False)


def startNFINDR(context, pca_data, maxit, checkbox_ATGP):
	'''
	Main function for N-Finder algorithm
	'''

	context.datasetAsArray = context.dataset.ReadAsArray()
	nfindr = NFindrModule()
	context.nfindr_data, Et, IDX, n_iterations = nfindr.NFINDR(data=pca_data, q=end_member_list[2], maxit=maxit, ATGP_init=checkbox_ATGP)
	context.logs.addItem("Analysis completed")
	context.logs.addItem(f'Number of iterations: {n_iterations}')
	context.logs.addItem("Generating Output file")
	writeData(context, "NFinder_", context.nfindr_data)
	context.logs.addItem(f"Output file NFinder_{OUTPUT_FILENAME} generated")
	setProgressBar(context, False)


def startSUNSAL(context, nfindr_data, iterations, lambda_val, tolerance, positivity, addOne, verbose):
	'''
	Main function for SUNSAL algorithm
	'''

	ss = SUNSALModule()
	context.logs.addItem("Initiating SUNSAL algorithm")
	context.sunsal_data, res_p, res_d, sunsal_i = ss.SUNSAL(M=np.transpose(nfindr_data), y=np.transpose(context.pca_data), al_iters=iterations, lambda_p=lambda_val, positivity=positivity, addone=addOne, tol=tolerance, verbose=verbose)
	context.logs.addItem("Running SUNSAL algorithm")
	writeData(context, "SUNSAL_", context.sunsal_data)
	context.logs.addItem(f"Output file SUNSAL_{OUTPUT_FILENAME} generated")
	context.logs.addItem(f"Number of iterations are {sunsal_i}")
	setProgressBar(context, False)


def startHfcVd(context):
	'''
	Main function for HfcVd algorithm
	'''
	context.datasetAsArray = np.array(context.dataset)
	context.logs.addItem("Initiating HfcVd algorithm")
	end_member_list = vd.HfcVd(context.datasetAsArray)
	context.logs.addItem("Running HfcVd algorithm")
	context.logs.addItem(f"Number of end member(s) found is/are {end_member_list[2]}")
	setProgressBar(context, False)


def startVCA(context, nfindr_data, SNR, verbose):
	'''
	Main function for VCA algorithm
	'''

	context.logs.addItem("Initiating VCA algorithm")
	context.vca_data, IDX, proj_data = sparse.vca(Y=nfindr_data, R=end_member_list[2], snr_input=SNR, verbose=verbose)
	context.logs.addItem("Running VCA algorithm")
	writeData(context, "VCA_", context.vca_data)
	context.logs.addItem(f"Output file VCA_{OUTPUT_FILENAME} generated")
	setProgressBar(context, False)


def startPPI(context, pca_data, skewers):
	'''
	Main function for PPI algorithm
	'''

	context.logs.addItem("Initiating PPI algorithm")
	context.ppi_data, IDX = eea.PPI(M=np.transpose(pca_data), q=end_member_list[2], numSkewers=skewers)
	context.logs.addItem("Running PPI algorithm")
	writeData(context, "PPI_", context.ppi_data)
	context.logs.addItem(f"Output file PPI_{OUTPUT_FILENAME} generated")
	setProgressBar(context, False)


def startNNLS(context, pca_data, nfindr_data):
	'''
	Main function for NNLS algorithm
	'''

	context.logs.addItem("Initiating NNLS algorithm")
	context.NNLS_data = LMM.NNLS(pca_data, nfindr_data)
	context.logs.addItem("Analysis completed")
	context.logs.addItem("Generating output file")
	writeData(context, "NNLS_", context.NNLS_data)
	context.logs.addItem(f"Output File NNLS_{OUTPUT_FILENAME} generated")
	setProgressBar(context, False)


def startUCLS(context, pca_data, nfindr_data):
	'''
	Main function for UCLS algorithm
	'''

	context.logs.addItem("Initiating UCLS algorithm")
	context.UCLS_data = LMM.UCLS(pca_data, nfindr_data)
	context.logs.addItem("Analysis completed")
	context.logs.addItem("Generating output file")
	writeData(context, "UCLS_", context.UCLS_data)
	context.logs.addItem(f"Output File UCLS_{OUTPUT_FILENAME} generated")
	setProgressBar(context, False)


def startFCLS(context, pca_data, nfindr_data):
	'''
	Main function for FCLS algorithm
	'''

	context.logs.addItem("Initiating FCLS algorithm")
	context.UCLS_data = LMM.FCLS(pca_data, nfindr_data)
	context.logs.addItem("Analysis completed")
	context.logs.addItem("Generating output file")
	writeData(context, "FCLS_", context.UCLS_data)
	context.logs.addItem(f"Output File FCLS_{OUTPUT_FILENAME} generated")
	setProgressBar(context, False)


def startGBMGDA(context, pca_data, nfindr_data):
	'''
	Main function to run GBM using semi NMF
	'''

	context.logs.addItem("Initiating GBM using Gradient analysis algorithm")
	context.GBMGDA_data = GBM_GDA.GBM_gradient(np.transpose(pca_data), np.transpose(nfindr_data))
	context.logs.addItem("Analysis completed")
	context.logs.addItem(f"RMS Error: {rmse}")
	context.logs.addItem("Generating output file")
	writeData(context, "GBMGDA_", context.GBMsemiNMF_data)
	context.logs.addItem(f"Output File GBMGDA_{OUTPUT_FILENAME} generated")
	context.setProgressBar(False)

def startGBMsemiNMF(context, pca_data, nfindr_data):
	'''
	Main function to run GBM using Gradient analysis
	'''
	context.logs.addItem("Initiating GBM using semiNMF algorithm")
	context.GBMsemiNMF_data, gamma = GBM_semiNMF.GBM_semiNMF(np.transpose(pca_data), np.transpose(nfindr_data))
	context.logs.addItem("Analysis completed")
	context.logs.addItem(f"RMS Error: {rmse}")
	context.logs.addItem("Generating output file")
	writeData(context, "GBMsemiNMF_", context.GBMsemiNMF_data)
	context.logs.addItem(f"Output File GBMsemiNMF_{OUTPUT_FILENAME} generated")
	context.setProgressBar(False)



def startPCA(context, selectedComponents):
	'''
	Main function for PCA
	'''

	# t1 = Thread(target=PCAThread2)
	context.datasetAsArray = context.dataset.ReadAsArray()
	pca = PrincipalComponentAnalysis(context.datasetAsArray)
	pca.scaleData()

	# t1.start()
	context.pca_data = pca.getPrincipalComponents_noOfComponents((int)(context.components.toPlainText()))
	retainedVariance = pca.getRetainedVariance((int)(context.components.toPlainText()))
	
	context.logs.addItem("Analysis completed")
	context.logs.addItem("Generating Output file")
	writeData(context, "PCA_", context.pca_data)
	# t1.join()
	context.logs.addItem(f"Output file PCA_{OUTPUT_FILENAME} generated")
	context.logs.addItem(f'Retained Variance: {retainedVariance}')
	setProgressBar(context,False)
	
	''' To plot the points after PCA '''
	if (int)(selectedComponents) == 1:
		newpid = os.fork()
		if newpid == 0:
			plot1DGraph(context, context.pca_data)

	elif (int)(selectedComponents) == 2:
		newpid = os.fork()
		if newpid == 0:
			plot2DGraph(context, context.pca_data)

	elif (int)(selectedComponents) == 3:
		newpid = os.fork()
		if newpid == 0:
			plot3DGraph(context, context.pca_data)

	else:
		context.logs.addItem('Due to high dimentionality, graph could not be plotted')


def startKerPCA(context, selectedComponents, jobs, kernel, solver, alpha, gamma, fit_inverse_transform, remove_zero_eigen):
	'''
	Main function for Kernel PCA
	'''

	typeCastedJobs = 0
	if jobs.toPlainText() == "" or not is_number(jobs.toPlainText()):
		typeCastedJobs = -1
	else:
		typeCastedJobs = (int)(jobs.toPlainText())

	typeCastedAlpha = 0
	if alpha.toPlainText() == "" or not is_number(alpha.toPlainText()):
		typeCastedAlpha = 1
	else:
		typeCastedAlpha = (int)(alpha.toPlainText())

	typeCastedGamma = 0
	if gamma.toPlainText() == "" or not is_number(gamma.toPlainText()):
		typeCastedGamma = 1/(int)(context.components.toPlainText())
	else:
		typeCastedGamma = (float)(gamma.toPlainText())


	context.datasetAsArray = context.dataset.ReadAsArray()
	kernelpca = KernelPCAAlgorithm(context.datasetAsArray, typeCastedJobs)
	kernelpca.scaleData()

	context.ker_pca_data = kernelpca.getPrincipalComponents_noOfComponents((int)(context.components.toPlainText()), typeCastedJobs, kernel, solver, typeCastedAlpha, typeCastedGamma, fit_inverse_transform, remove_zero_eigen)
	
	context.logs.addItem("Analysis completed")
	context.logs.addItem("Generating Output file")
	writeData(context, "KernelPCA_", context.ker_pca_data)
	context.logs.addItem(f"Output file KernelPCA_{OUTPUT_FILENAME} generated")
	
	''' To plot the points after Kernel PCA '''
	if (int)(selectedComponents) == 1:
		newpid = os.fork()
		if newpid == 0:
			plot1DGraph(context, context.ker_pca_data)

	elif (int)(selectedComponents) == 2:
		newpid = os.fork()
		if newpid == 0:
			plot2DGraph(context, context.ker_pca_data)

	elif (int)(selectedComponents) == 3:
		newpid = os.fork()
		if newpid == 0:
			plot3DGraph(context, context.ker_pca_data)

	else:
		context.logs.addItem('Due to high dimentionality, graph could not be plotted')


def writeData(context, prefix, data):
	'''
	Writes data into a file in CSV (Comma Seperated Value) format
	'''

	with open(prefix + OUTPUT_FILENAME, 'w') as writeFile:
		writer = csv.writer(writeFile)

		dataList = []
		for row in data:
			temp = []
			for cols in row:
				temp.append(cols)
			dataList.append(temp)
		writer.writerows(dataList)

	writeFile.close()


def plot1DGraph(context, data):
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


def plot2DGraph(context, data):
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


def plot3DGraph(context, data):
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


def writeError(context, err_msg):
	'''
	This method receives input from stderr as PyQtSlot and prints it in the 
	logs section
	'''

	context.logs.addItem(err_msg)


##########################################
###### Tabs for different algorithms #####
##########################################

def startPCAWindow(context):
	clearWidgets(context)
	context.input_label.hide()
	context.ToolTab = PCAUI()
	context.setWindowTitle("Principal Component Analysis")
	context.setCentralWidget(context.ToolTab)
	context.show()	

def startKerPCAWindow(context):
	clearWidgets(context)
	context.input_label.hide()
	context.ToolTab = KerPCAUI()
	context.setWindowTitle("Kernel PCA")
	context.setCentralWidget(context.ToolTab)
	context.show()

def startNFINDRWindow(context):
	clearWidgets(context)
	context.input_label.hide()
	context.ToolTab = NFINDRUI()
	context.setWindowTitle("NFINDR")
	context.setCentralWidget(context.ToolTab)
	context.show()

def startLLEWindow(context):
	clearWidgets(context)
	context.input_label.hide()
	context.ToolTab = LLEUI()
	context.setWindowTitle("Locally Linear Embedding")
	context.setCentralWidget(context.ToolTab)
	context.show()

def startSUNSALWindow(context):
	clearWidgets(context)
	context.input_label.hide()
	context.ToolTab = SUNSALUI()
	context.setWindowTitle("SUNSAL")
	context.setCentralWidget(context.ToolTab)
	context.show()

def startHFCVDWindow(context):
	clearWidgets(context)
	context.input_label.hide()
	context.ToolTab = HFCVDUI()
	context.setWindowTitle("HFCVD")
	context.setCentralWidget(context.ToolTab)
	context.show()

def startATGPWindow(context):
	clearWidgets(context)
	context.input_label.hide()
	context.ToolTab = ATGPUI()
	context.setWindowTitle("ATGP")
	context.setCentralWidget(context.ToolTab)
	context.show()

def startPPIWindow(context):
	clearWidgets(context)
	context.input_label.hide()
	context.ToolTab = PPIUI()
	context.setWindowTitle("PPI")
	context.setCentralWidget(context.ToolTab)
	context.show()

def startVCAWindow(context):
	clearWidgets(context)
	context.input_label.hide()
	context.ToolTab = VCAUI()
	context.setWindowTitle("Vertex Component Analysis")
	context.setCentralWidget(context.ToolTab)
	context.show()

def startNMFWindow(context):
	clearWidgets(context)
	context.input_label.hide()
	context.ToolTab = NMFUI()
	context.setWindowTitle("Non-negative Matrix Factorization")
	context.setCentralWidget(context.ToolTab)
	context.show()	

def startNNLSWindow(context):
	clearWidgets(context)
	context.input_label.hide()
	context.ToolTab = NNLSUI()
	context.setWindowTitle("Non-negative Constrained Least Square Abundance Estimation")
	context.setCentralWidget(context.ToolTab)
	context.show()	

def startUCLSWindow(context):
	clearWidgets(context)
	context.input_label.hide()
	context.ToolTab = NNLSUI()
	context.setWindowTitle("Unconstrained Least Squares Abundance Estimation")
	context.setCentralWidget(context.ToolTab)
	context.show()	

def startFCLSWindow(context):
	clearWidgets(context)
	context.input_label.hide()
	context.ToolTab = NNLSUI()
	context.setWindowTitle("Fully Constrained Least Squares Abundance Estimation")
	context.setCentralWidget(context.ToolTab)
	context.show()

def startGBMsemiNMFWindow(context):
	clearWidgets(context)
	context.input_label.hide()
	context.ToolTab = GBMsemiNMFUI()
	context.setWindowTitle("Generalized Bilinear Model using Semi-NMF")
	context.setCentralWidget(context.ToolTab)
	context.show()	

def startGBMGDAWindow(context):
	clearWidgets(context)
	context.input_label.hide()
	context.ToolTab = NNLSUI()
	context.setWindowTitle("Generalized Bilinear Model using Gradient analysis")
	context.setCentralWidget(context.ToolTab)
	context.show()

def clearWidgets(context):

    context.input_label.hide()
    context.input_browse.hide()
    context.input_text.hide()
    context.output_label.hide()
    context.output_browse.hide()
    context.output_text.hide()
    context.components_label.hide()
    context.components.hide()
    context.jobs_label.hide()
    context.jobs.hide()
    context.OK.hide()
    context.cancel.hide()
    context.logs.hide()
    context.progress.hide()


if __name__ == "__main__":

	app = QApplication(sys.argv)
	window = Software()
	window.show()

	# Adding error handler
	#std_err_handler = StdErrHandler()
	#sys.stderr = std_err_handler
	#std_err_handler.err_msg.connect(window.writeError)
sys.exit(app.exec_())