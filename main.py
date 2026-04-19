#import cupy as cp
import numpy as np
import OpenGL.GL as gl
from OpenGL import GLU
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtOpenGL
import sys

class MainWindow(QtGui.QWindow):
    def __init__(self):
        QtGui.QWindow.__init__(self)
        self.resize(600, 600)
        self.setWindowTitle("Quantum Simulation")


if __name__ == '__main__':
    app = QtGui.QGuiApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())