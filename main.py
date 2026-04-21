import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL import GLU
from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6 import QtOpenGL
from PyQt6.QtCore import QTimer, Qt
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

out vec4 FragColor;

uniform vec2 center;
uniform vec2 screen;
uniform float resolution;
uniform float scale;

void main() {
    vec2 uv = gl_FragCoord.xy * scale / resolution + screen;
    
    float d = length(uv - center);

    if (d < 0.1)
        FragColor = vec4(1, 0, 0, 1);
    else
        FragColor = vec4(0, 0, 0, 1);
}
"""


class GLWidget(QOpenGLWidget):
    def __init__(self, w, h):
        super().__init__()
        self.w = w
        self.h = h
        self.pixel_data = xp.zeros((self.w, self.h, 4), dtype=np.uint8)
        self.pixel_data[:, :, 3] = 255
        self.pixel_data_OG = xp.copy(self.pixel_data)
        self.center = xp.array([0, 0], dtype=xp.float32)
        self.screen_corner = xp.array([0, 0], dtype=xp.float32)
        self.scale = 1
        self.old_scale = 1

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_sim)
        self.timer.start(16)

        self.keys = set()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.mouse_down = False
        self.prev_mouse_coords = xp.array([0, 0], dtype=xp.float32)
        self.prev_screen_corner = xp.array([0, 0], dtype=xp.float32)


        ratio = self.h/self.w
        width_pixels = xp.linspace(-10, 10, w)
        height_pixels = xp.linspace(-10*ratio, 10*ratio, h)
        A, B = xp.meshgrid(width_pixels, height_pixels)
        self.pixel_pos = xp.stack([A, B], axis=-1)

    def keyPressEvent(self, event):
        self.keys.add(event.key())

    def keyReleaseEvent(self, event):
        self.keys.discard(event.key())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_down = True
            self.prev_mouse_coords[0] = event.position().x()
            self.prev_mouse_coords[1] = event.position().y()
            self.prev_screen_corner[0] = self.screen_corner[0]
            self.prev_screen_corner[1] = self.screen_corner[1]

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_down = False

    def mouseMoveEvent(self, event):
        self.scrolling = False
        if self.mouse_down:
            self.screen_corner[0] = self.prev_screen_corner[0] - 2*(event.position().x() - self.prev_mouse_coords[0])*self.scale/(self.h)
            self.screen_corner[1] = self.prev_screen_corner[1] + 2*(event.position().y() - self.prev_mouse_coords[1])*self.scale/(self.h)

    def wheelEvent(self, event):
        self.scrolling = True
        delta = event.angleDelta().y()
        factor = 1.2
        mouse_pos = (self.screen_corner + 2*xp.array([event.position().x()/self.w, 1 - event.position().y()/self.h], dtype=xp.float32)*self.scale) #*xp.array([self.w/self.h, 1.0])
        self.old_scale = xp.copy(self.scale)
        if delta > 0 and self.scale < 8:
            self.scale *= factor # smaller

        elif delta < 0 and self.scale > 1/8:
            self.scale /= factor # bigger

        vec = mouse_pos - self.screen_corner
        self.screen_corner += vec*(1 - self.scale/self.old_scale)
        print(mouse_pos)

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
        quad = xp.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=xp.float32)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.w, self.h,0, GL_RGBA, GL_UNSIGNED_BYTE,None)

    def update_sim(self):
        if Qt.Key.Key_A in self.keys:
            self.screen_corner[0] -= 0.01

        if Qt.Key.Key_D in self.keys:
            self.screen_corner[0] += 0.01

        if Qt.Key.Key_W in self.keys:
            self.screen_corner[1] += 0.01

        if Qt.Key.Key_S in self.keys:
            self.screen_corner[1] -= 0.01


        self.center[0] = 1.5
        self.center[1] = 1
        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.shader)

        center_loc = glGetUniformLocation(self.shader, "center")
        glUniform2f(center_loc, self.center[0], self.center[1])

        screen_loc = glGetUniformLocation(self.shader, "screen")
        glUniform2f(screen_loc, self.screen_corner[0], self.screen_corner[1])

        scale_loc = glGetUniformLocation(self.shader, "scale")
        glUniform1f(scale_loc, self.scale)

        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform1f(res_loc, self.h)

        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        screen = QApplication.primaryScreen() or QApplication.screens()[0]
        rect = screen.availableGeometry()

        self.screen_w = rect.width()
        self.screen_h = rect.height()
        #self.screen_w = 800
        #self.screen_h = 800

        widget = GLWidget(self.screen_w, self.screen_h)
        self.setCentralWidget(widget)

        self.showMaximized()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()