from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal

class ValidationThread(QThread):

	startThread = pyqtSignal()
	startProgress = pyqtSignal(bool)

	def __init__(self):
		QThread.__init__(self)

	def __del__(self):
		self.wait()

	def run(self):
		self.startProgress.emit(True)
		self.startThread.emit()
