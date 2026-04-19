#import cupy as cp
import numpy as np
from OpenGL.GL import *
from OpenGL import GLU
from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6 import QtOpenGL
from PyQt6.QtCore import QTimer
import sys

xp = np

class GLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.w = 800
        self.h = 800
        self.pixel_data = xp.zeros((self.w, self.h, 4), dtype=np.uint8)
        self.center = xp.zeros(2)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_sim)
        self.timer.start(16)

    def initializeGL(self):
        glClearColor(0, 0, 0, 1)

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

