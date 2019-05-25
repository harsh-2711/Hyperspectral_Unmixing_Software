from PyQt5 import QtCore

class StdErrHandler(QtCore.QObject):

	err_msg = QtCore.pyqtSignal(str)

	def __init__(self):
		QtCore.QObject.__init__(self)

	def write(self, msg):
		self.err_msg.emit(msg)