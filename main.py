#import cupy as cp
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL import GLU
from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6 import QtOpenGL
from PyQt6.QtCore import QTimer
import sys

xp = np


VERTEX_SHADER = """
#version 330 core

layout (location = 0) in vec2 position;
out vec2 uv;

void main() {
    uv = (position + 1.0) * 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core

in vec2 uv;
out vec4 FragColor;

uniform vec2 center;

void main() {
    float d = length(uv - center);

    if (d < 0.1)
        FragColor = vec4(1, 0, 0, 1);
    else
        FragColor = vec4(0, 0, 0, 1);
}
"""

quad = xp.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=xp.float32)

class GLWidget(QOpenGLWidget):
    def __init__(self, w, h):
        super().__init__()
        self.w = w
        self.h = h
        self.pixel_data = xp.zeros((self.w, self.h, 4), dtype=np.uint8)
        self.pixel_data[:, :, 3] = 255
        self.pixel_data_OG = xp.copy(self.pixel_data)
        self.center = xp.array([0, 0.5], dtype=xp.float32)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_sim)
        self.timer.start(16)

        width_pixels = xp.linspace(-10, 10, w)
        height_pixels = xp.linspace(-10, 10, h)
        A, B = xp.meshgrid(width_pixels, height_pixels)
        self.pixel_pos = xp.stack([A, B], axis=-1)

    def initializeGL(self):
        glClearColor(0, 0, 0, 1)

        self.shader = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )

        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)

        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.w, self.h,0, GL_RGBA, GL_UNSIGNED_BYTE,None)

    def update_sim(self):
        self.center[0] += 0.01
        self.update()
        '''
        self.pixel_data = xp.copy(self.pixel_data_OG)
        self.center[0] += 1
        mask = (xp.sqrt(xp.square(self.pixel_pos[:, :, 0] - self.center[0]) + xp.square(self.pixel_pos[:, :, 1] - self.center[1])) <= 4)
        self.pixel_data[:, :, 0][mask] = 255
        self.update()
        '''
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUseProgram(self.shader)
        center_loc = glGetUniformLocation(self.shader, "center")
        glUniform2f(center_loc, self.center[0], self.center[1])
        glTexSubImage2D(GL_TEXTURE_2D,0,0, 0, self.w, self.h, GL_RGBA, GL_UNSIGNED_BYTE, self.pixel_data)
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(-1, -1)
        glTexCoord2f(1, 0)
        glVertex2f(1, -1)
        glTexCoord2f(1, 1)
        glVertex2f(1, 1)
        glTexCoord2f(0, 1)
        glVertex2f(-1, 1)
        glEnd()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        screen = QApplication.primaryScreen() or QApplication.screens()[0]
        rect = screen.availableGeometry()

        #self.screen_w = rect.width()
        #self.screen_h = rect.height()
        self.screen_w = 800
        self.screen_h = 800

        widget = GLWidget(self.screen_w, self.screen_h)
        self.setCentralWidget(widget)

        self.showMaximized()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()