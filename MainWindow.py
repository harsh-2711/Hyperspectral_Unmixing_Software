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
from Modules.Non_Linear_Unmixing import GBM_semiNMF

from Threads import ValidationThread
from threading import Thread
import subprocess
from ErrorHandler import StdErrHandler

import multiprocessing as mp

import matplotlib.pyplot as plt


path = os.path.dirname(__file__)
qtCreatorFile = "MainWindow.ui" 

Ui_MainWindow, QtBaseClass = uic.loadUiType(path + qtCreatorFile)


class KerPCAUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(KerPCAUI, self).__init__()

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

class NFINDRUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(NFINDRUI, self).__init__()


		# Input Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 30)

		# Input browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,30)

		# Input text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,35,402,21)

		# Transform Input Label
		self.output_label = QLabel("Transform I/P", self)
		self.output_label.move(20, 85)

		# Transform Input browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,85)

		# Transform Input text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,90,402,21)
		self.output_text.setText(os.getcwd())


		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 140)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,140)

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,145,402,21)
		self.output_text.setText(os.getcwd())

		# No of endmembers Label
		self.components_label = QLabel("Endmembers", self)
		self.components_label.move(20, 200)

		# Endmembers text field
		self.components = QTextEdit(self)
		self.components.setGeometry(140,205,45,21)

		# Max iterations Label
		self.jobs_label = QLabel("Max iterations", self)
		self.jobs_label.move(220, 200)

		# Max iterations text field
		self.jobs = QTextEdit(self)
		self.jobs.setGeometry(350,205,40,21)

		# ATGP label
		self.fit_inverse_transform = QLabel("ATGP", self)
		self.fit_inverse_transform.setGeometry(440,208,280,15)
		
		# ATGP Checkbox
		self.checkbox_fit_inverse_transform = QCheckBox(self)
		self.checkbox_fit_inverse_transform.move(500, 202)

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


class LLEUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(LLEUI, self).__init__()


		# Input Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 20)

		# Input browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,20)

		# Input text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,25,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 70)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,70)

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,75,402,21)
		self.output_text.setText(os.getcwd())

		# No of endmembers Label
		self.components_label = QLabel("Neighbours", self)
		self.components_label.move(20, 120)

		# Endmembers text field
		self.components = QTextEdit(self)
		self.components.setGeometry(140,125,45,21)

		# Components Label
		self.jobs_label = QLabel("Components", self)
		self.jobs_label.move(240, 120)

		# Components text field
		self.jobs = QTextEdit(self)
		self.jobs.setGeometry(340,125,40,21)

		# Jobs Label
		self.jobs_label = QLabel("Jobs", self)
		self.jobs_label.move(450, 120)

		# Jobs text field
		self.jobs = QTextEdit(self)
		self.jobs.setGeometry(510,125,40,21)

		# Eigen solver Label
		self.kernel_label = QLabel("Eigen solver", self)
		self.kernel_label.move(140, 180)

		# Eigen Solver Choice List
		self.kernelChoiceList = QComboBox(self)
		self.kernelChoiceList.addItem("auto")
		self.kernelChoiceList.addItem("arpack")	
		self.kernelChoiceList.addItem("dense")
		self.kernelChoiceList.move(230, 180)

		# Method Label
		self.kernel_label = QLabel("Method", self)
		self.kernel_label.move(400, 180)

		# Method Choice List
		self.kernelChoiceList = QComboBox(self)
		self.kernelChoiceList.addItem("standard")
		self.kernelChoiceList.addItem("hessian")
		self.kernelChoiceList.addItem("modified")
		self.kernelChoiceList.addItem("ltsa")
		self.kernelChoiceList.move(465, 180)

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


class SUNSALUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(SUNSALUI, self).__init__()


		# Endmember sign Label
		self.input_label = QLabel("EM signature", self)
		self.input_label.move(20, 20)

		# Endmember sign browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,20)

		# Endmember sign text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,25,402,21)

		# Data matrix Label
		self.output_label = QLabel("Data Matrix", self)
		self.output_label.move(20, 70)

		# Data Matrix browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,70)

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

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,125,402,21)
		self.output_text.setText(os.getcwd())

		# No of iterations Label
		self.components_label = QLabel("Min AL iter", self)
		self.components_label.move(60, 170)

		# iterations text field
		self.components = QTextEdit(self)
		self.components.setGeometry(140,175,45,21)

		# Components Label
		self.jobs_label = QLabel("Lambda", self)
		self.jobs_label.move(260, 170)

		# Components text field
		self.jobs = QTextEdit(self)
		self.jobs.setGeometry(320,175,40,21)

		# Tolerance Label
		self.jobs_label = QLabel("Tolerance", self)
		self.jobs_label.move(430, 170)

		# Tolerance text field
		self.jobs = QTextEdit(self)
		self.jobs.setGeometry(510,175,40,21)


		# Positivity label
		self.fit_inverse_transform = QLabel("Positivity", self)
		self.fit_inverse_transform.setGeometry(130,212,280,15)
		
		# Positivity Checkbox
		self.checkbox_fit_inverse_transform = QCheckBox(self)
		self.checkbox_fit_inverse_transform.move(200, 206)

		# Addone label
		self.fit_inverse_transform = QLabel("Add one", self)
		self.fit_inverse_transform.setGeometry(250,212,280,15)
		
		# Addone Checkbox
		self.checkbox_fit_inverse_transform = QCheckBox(self)
		self.checkbox_fit_inverse_transform.move(320, 206)

		# Verbose label
		self.fit_inverse_transform = QLabel("Verbose", self)
		self.fit_inverse_transform.setGeometry(370,212,280,15)
		
		# Verbose Checkbox
		self.checkbox_fit_inverse_transform = QCheckBox(self)
		self.checkbox_fit_inverse_transform.move(440, 206)

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


class HFCVDUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(HFCVDUI, self).__init__()


		# HSI Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 60)

		# HSI browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,60)

		# HSI text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,65,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 120)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,120)

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,125,402,21)
		self.output_text.setText(os.getcwd())

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


class NMFUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(NMFUI, self).__init__()


		# HSI Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 20)

		# HSI browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,20)

		# HSI text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,25,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 70)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550, 70)

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


		# Method Label
		self.solver_label = QLabel("Solver", self)
		self.solver_label.move(300, 180)

		# Method Choice List
		self.solverChoiceList = QComboBox(self)
		self.solverChoiceList.addItem("Coordinate Descent")
		self.solverChoiceList.addItem("Multiplicative Update")
		self.solverChoiceList.move(365, 180)

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


class PPIUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(PPIUI, self).__init__()


		# Endmember sign Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 20)

		# Endmember sign browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,20)

		# Endmember sign text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,25,402,21)

		# Data matrix Label
		self.output_label = QLabel("Init Skewers", self)
		self.output_label.move(20, 70)

		# Data Matrix browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,70)

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

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,125,402,21)
		self.output_text.setText(os.getcwd())

		# Tolerance Label		
		self.jobs_label = QLabel("No of Skewers", self)
		self.jobs_label.move(150, 180)

		# Tolerance text field
		self.jobs = QTextEdit(self)
		self.jobs.setGeometry(280,185,40,21)

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

class VCAUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(VCAUI, self).__init__()


		# HSI Label
		self.input_label = QLabel("HSI Input", self)
		self.input_label.move(20, 45)

		# HSI browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,45)

		# HSI text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,50,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 100)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,100)

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,105,402,21)
		self.output_text.setText(os.getcwd())

		self.jobs_label = QLabel("Endmembers", self)
		self.jobs_label.move(150, 150)

		# Tolerance text field
		self.jobs = QTextEdit(self)
		self.jobs.setGeometry(260,155,40,21)

		self.jobs_label = QLabel("SNR input", self)
		self.jobs_label.move(340, 150)

		# Tolerance text field
		self.jobs = QTextEdit(self)
		self.jobs.setGeometry(420,155,40,21)

		self.fit_inverse_transform = QLabel("Verbose", self)
		self.fit_inverse_transform.setGeometry(150,212,280,15)
		
		# Verbose Checkbox
		self.checkbox_fit_inverse_transform = QCheckBox(self)
		self.checkbox_fit_inverse_transform.move(220, 206)

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

class ATGPUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(ATGPUI, self).__init__()


		# HSI Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 60)

		# HSI browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,60)

		# HSI text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,65,402,21)

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 120)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,120)

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,125,402,21)
		self.output_text.setText(os.getcwd())

		# Tolerance Label		
		self.jobs_label = QLabel("No of Endmembers", self)
		self.jobs_label.move(150, 180)

		# Tolerance text field
		self.jobs = QTextEdit(self)
		self.jobs.setGeometry(280,185,40,21)


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


class NNLSUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(NNLSUI, self).__init__()


		# Endmember sign Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 40)

		# Endmember sign browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,40)

		# Endmember sign text field
		self.input_text = QTextEdit(self)
		self.input_text.setGeometry(142,45,402,21)

		# Data matrix Label
		self.output_label = QLabel("Endmember Mat", self)
		self.output_label.move(20, 95)

		# Data Matrix browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,95)

		# Data Matrix text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,100,402,21)
		self.output_text.setText(os.getcwd())

		# Output Label
		self.output_label = QLabel("Output", self)
		self.output_label.move(20, 150)

		# Output browse button
		self.output_browse = QPushButton("Browse", self)
		self.output_browse.move(550,150)

		# Output text field
		self.output_text = QTextEdit(self)
		self.output_text.setGeometry(142,155,402,21)
		self.output_text.setText(os.getcwd())

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


class GBMsemiNMFUI(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super(GBMsemiNMFUI, self).__init__()


		# Endmember sign Label
		self.input_label = QLabel("Input", self)
		self.input_label.move(20, 20)

		# Endmember sign browse button
		self.input_browse = QPushButton("Browse", self)
		self.input_browse.move(550,20)

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
		pcaMenu.triggered.connect(partial(self.changeCurrentAlgo, "PCA"))

		nmf = QAction("NMF", self)
		dimReduction.addAction(nmf)
		nmf.triggered.connect(partial(startNMFWindow, self))
		nmf.triggered.connect(partial(self.changeCurrentAlgo, "NMF"))

		kerPCA = QAction("Kernel PCA", self)
		dimReduction.addAction(kerPCA)
		kerPCA.triggered.connect(partial(startKerPCAWindow, self))
		kerPCA.triggered.connect(partial(self.changeCurrentAlgo, "KerPCA"))

		lda = QAction("FDA", self)
		dimReduction.addAction(lda) 

		lle = QAction("LLE", self)
		dimReduction.addAction(lle)
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
		hfcvd.triggered.connect(partial(startHFCVDWindow, self))
		hfcvd.triggered.connect(partial(self.changeCurrentAlgo, "HfcVd"))

		# End Member Extraction
		eme = menubar.addMenu("End Member Extraction")

		nFinder = QAction("N-Finder", self)
		eme.addAction(nFinder)
		nFinder.triggered.connect(partial(startNFINDRWindow, self))
		nFinder.triggered.connect(partial(self.changeCurrentAlgo, "NFinder"))

		atgp = QAction("ATGP", self)
		eme.addAction(atgp)
		atgp.triggered.connect(partial(startATGPWindow, self))
		atgp.triggered.connect(partial(self.changeCurrentAlgo, "ATGP"))

		ppi = QAction("PPI", self)
		eme.addAction(ppi)
		ppi.triggered.connect(partial(startPPIWindow, self))
		ppi.triggered.connect(partial(self.changeCurrentAlgo, "PPI"))

		sisal = QAction("SISAL", self)
		eme.addAction(sisal)
		sisal.triggered.connect(partial(self.changeCurrentAlgo, "SISAL"))

		# Linear Unmixing
		lu = menubar.addMenu("Linear Unmixing")

		sunsal = QAction("SUNSAL", self)
		lu.addAction(sunsal)
		sunsal.triggered.connect(partial(startSUNSALWindow, self))
		sunsal.triggered.connect(partial(self.changeCurrentAlgo, "SUNSAL"))

		vca = QAction("VCA", self)
		lu.addAction(vca)
		vca.triggered.connect(partial(startVCAWindow, self))
		vca.triggered.connect(partial(self.changeCurrentAlgo, "VCA"))

		nnls = QAction("NNLS", self)
		lu.addAction(nnls)
		nnls.triggered.connect(partial(startNNLSWindow, self))
		nnls.triggered.connect(partial(self.changeCurrentAlgo, "NNLS"))

		ucls = QAction("UCLS", self)
		lu.addAction(ucls)
		ucls.triggered.connect(partial(startUCLSWindow, self))
		ucls.triggered.connect(partial(self.changeCurrentAlgo, "UCLS"))

		fcls = QAction("FCLS", self)
		lu.addAction(fcls)
		fcls.triggered.connect(partial(startFCLSWindow, self))
		fcls.triggered.connect(partial(self.changeCurrentAlgo, "FCLS"))

		# Non-linear Unmixing
		nlu = menubar.addMenu("Non Linear Unmixing")

		gbmNMF = QAction("GBM using semi-NMF", self)
		nlu.addAction(gbmNMF)
		gbmNMF.triggered.connect(partial(startGBMsemiNMFWindow, self))
		gbmNMF.triggered.connect(partial(self.changeCurrentAlgo, "GBM using semi-NMF"))

		gbmGrad = QAction("GBM using gradient", self)
		nlu.addAction(gbmGrad)
		gbmGrad.triggered.connect(partial(self.changeCurrentAlgo, "GBM using gradient"))

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
		self.tolerance.setText("")
		self.maxit.setText("")
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
		tolerance = self.tolerance.toPlainText()
		max_iterations = self.maxit.toPlainText()

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
					nmf_data = self.startNMF(selectedComponents, tolerance, max_iterations)

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

			elif self.currentAlgo == "PPI":
				self.logs.addItem(f'Starting PPI for getting Endmember Extraction')
				self.startHfcVd(selectedComponents)
				self.startPPI(self.pca_data)

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

			elif self.currentAlgo == "GBM using semi-NMF":
				self.logs.addItem(f'Starting Generalized Bilinear Model for Non-Linear Unmixing')
				self.startHfcVd(selectedComponents)
				self.startNFINDR(self.pca_data)
				self.startGBMsemiNMF(self.pca_data, self.nfindr_data)

	
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



	def startNMF(self,selectedComponents, tolerance, max_iterations):
		'''
		Main function for NMF
		'''

		self.datasetAsArray = self.dataset.ReadAsArray()
		nmf = NonNegativeMatrixFactorisation(self.datasetAsArray)
		nmf.scaleData()
		self.nmf_data = nmf.getReducedComponents_noOfComponents((int)(self.components.toPlainText()), (int)(self.tolerance.toPlainText()), (int)(self.max_iterations.toPlainText()))
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


	def startGBMsemiNMF(self, pca_data, nfindr_data):
		'''
		Main function to run GBM using semi NMF
		'''

		self.logs.addItem("Initiating GBM using semiNMF algorithm")
		self.GBMsemiNMF_data, rmse = GBM_semiNMF.GBM_semiNMF(np.transpose(pca_data), np.transpose(nfindr_data))
		self.logs.addItem("Analysis completed")
		self.logs.addItem(f"RMS Error: {rmse}")
		self.logs.addItem("Generating output file")
		self.writeData("GBMsemiNMF_", self.GBMsemiNMF_data)
		self.logs.addItem(f"Output File GBMsemiNMF_{self.OUTPUT_FILENAME} generated")
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


##########################################
###### Tabs for different algorithms #####
##########################################

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