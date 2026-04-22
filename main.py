import cupy as cp
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

xp = cp


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
    def __init__(self):
        super().__init__()
        dpr = self.devicePixelRatioF()
        fb_w = self.width() * dpr
        fb_h = self.height() * dpr
        self.center = np.array([0, 0], dtype=xp.float32)
        self.screen_corner = np.array([0, 0], dtype=xp.float32)
        self.scale = 1
        self.old_scale = 1

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_sim)
        self.timer.start(16)

        self.keys = set()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.mouse_down = False
        self.prev_mouse_coords = np.array([0, 0], dtype=xp.float32)
        self.prev_screen_corner = np.array([0, 0], dtype=xp.float32)


        e_mass = 9.109e-31
        c = 3e8
        e_vel_x = 0.01*c
        e_vel_y = 0
        pi = 3.14159
        hbar = 1.05457e-34
        k_0_x = e_vel_x*e_mass/hbar
        k_0_y = e_vel_y*e_mass/hbar
        k_0 = xp.sqrt(k_0_x**2 + k_0_y**2)
        e_wavelength = 2*pi/k_0
        cell_spacing = e_wavelength/20
        sigma = 8*e_wavelength
        L = 12*sigma
        delta_t = e_mass*cell_spacing**2/(8*hbar)

        ratio = fb_h/fb_w
        width_pixels = xp.linspace(0, 10, 1000)
        height_pixels = xp.linspace(0, 10*ratio, 1000)
        A, B = xp.meshgrid(width_pixels, height_pixels)
        cells = xp.stack([A, B], axis=-1)

    def keyPressEvent(self, event):
        self.keys.add(event.key())

    def keyReleaseEvent(self, event):
        self.keys.discard(event.key())

    def mousePressEvent(self, event):
        dpr = self.devicePixelRatioF()
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_down = True
            self.prev_mouse_coords[0] = event.position().x() * dpr
            self.prev_mouse_coords[1] = event.position().y() * dpr
            self.prev_screen_corner[0] = self.screen_corner[0]
            self.prev_screen_corner[1] = self.screen_corner[1]

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_down = False

    def mouseMoveEvent(self, event):
        self.scrolling = False
        dpr = self.devicePixelRatioF()
        fb_h = self.height() * dpr
        if self.mouse_down:
            self.screen_corner[0] = self.prev_screen_corner[0] - (event.position().x() * dpr - self.prev_mouse_coords[0])*self.scale/fb_h
            self.screen_corner[1] = self.prev_screen_corner[1] + (event.position().y() * dpr - self.prev_mouse_coords[1])*self.scale/fb_h

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.1

        dpr = self.devicePixelRatioF()
        fb_h = self.height() * dpr

        mouse_screen = np.array([event.position().x() * dpr/fb_h, 1 - event.position().y() * dpr/fb_h], dtype=np.float32)
        mouse_pos = self.screen_corner + mouse_screen*self.scale
        self.old_scale = np.copy(self.scale)
        if delta > 0 and self.scale < 4:
            self.scale *= factor # smaller

        elif delta < 0 and self.scale > 1/4:
            self.scale /= factor # bigger

        self.screen_corner = mouse_pos - mouse_screen*self.scale
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
        quad = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        dpr = self.devicePixelRatioF()
        fb_h = self.height() * dpr
        fb_w = self.width() * dpr
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fb_w, fb_h,0, GL_RGBA, GL_UNSIGNED_BYTE,None)

    def update_sim(self):
        if Qt.Key.Key_A in self.keys:
            self.screen_corner[0] -= 0.01

        if Qt.Key.Key_D in self.keys:
            self.screen_corner[0] += 0.01

        if Qt.Key.Key_W in self.keys:
            self.screen_corner[1] += 0.01

        if Qt.Key.Key_S in self.keys:
            self.screen_corner[1] -= 0.01


        self.center[0] = 0
        self.center[1] = 1
        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader)

        dpr = self.devicePixelRatioF()
        fb_h = self.height() * dpr

        center_loc = glGetUniformLocation(self.shader, "center")
        glUniform2f(center_loc, self.center[0], self.center[1])

        screen_loc = glGetUniformLocation(self.shader, "screen")
        glUniform2f(screen_loc, self.screen_corner[0], self.screen_corner[1])

        scale_loc = glGetUniformLocation(self.shader, "scale")
        glUniform1f(scale_loc, self.scale)

        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform1f(res_loc, fb_h)

        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        widget = GLWidget()
        self.setCentralWidget(widget)

        self.showMaximized()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()