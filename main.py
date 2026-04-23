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
uniform vec2 max;
uniform float resolution;
uniform float scale;

void main() {
    vec2 uv = gl_FragCoord.xy * scale / resolution + screen;
    float padding = 5e-10;
    if (uv.x > 0 && uv.x < max.x && uv.y > 0 && uv.y < max.y)
        FragColor = vec4(1, 0, 0, 1);
    else if (uv.x > -padding && uv.x < max.x + padding && uv.y > -padding && uv.y < max.y + padding)
        FragColor = vec4(0.5, 0.5, 0.5, 1);
    else
        FragColor = vec4(0, 0, 0, 1);

}
"""


class GLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        dpr = self.devicePixelRatioF()
        fb_h = max(self.height() * dpr, 1.0)
        fb_w = max(self.width() * dpr, 1.0)
        self.center = np.array([0, 0], dtype=np.float32)
        self.screen_corner = np.array([0, 0], dtype=np.float32)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_sim)
        self.timer.start(16)

        self.keys = set()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.mouse_down = False
        self.prev_mouse_coords = np.array([0, 0], dtype=np.float32)
        self.prev_screen_corner = np.array([0, 0], dtype=np.float32)
        ratio = fb_h / fb_w


        self.e_mass = 9.109e-31
        c = 3e8
        e_vel_x = 0.01*c
        e_vel_y = 0
        self.hbar = 1.05457e-34
        k_0_x = e_vel_x*self.e_mass/self.hbar
        k_0_y = e_vel_y*self.e_mass/self.hbar
        self.k_0 = xp.sqrt(k_0_x**2 + k_0_y**2)
        e_wavelength = 2*xp.pi/self.k_0
        cell_spacing = e_wavelength/20
        sigma = 8*e_wavelength
        self.L = 12*sigma
        self.delta_t = self.e_mass*cell_spacing**2/(8*self.hbar)
        x_i = self.L/10
        y_i = self.L*ratio/2


        width_cells = xp.linspace(0, self.L, int(self.L/cell_spacing))
        height_cells = xp.linspace(0, self.L*ratio, int(self.L*ratio/cell_spacing))
        A, B = xp.meshgrid(width_cells, height_cells)
        cell_pos = xp.stack([A, B], axis=-1)

        # Initial Values
        self.psi = xp.exp((1j)*(k_0_x*(cell_pos[..., 0] - x_i) + k_0_y*(cell_pos[..., 1] - y_i)) - ((cell_pos[..., 0] - x_i)**2 + (cell_pos[..., 1] - y_i)**2)/(2*sigma**2))

        # Integrate
        psi_prob_int = (xp.abs(self.psi)**2)
        psi_prob_int = xp.sum(psi_prob_int)
        psi_prob_int *= cell_spacing**2

        # Normalize wavefunction
        self.psi /= xp.sqrt(psi_prob_int)

        self.V = xp.zeros_like(self.psi)

        self.scale = float(self.L*ratio)
        self.old_scale = float(self.L*ratio)
        self.initial_scale = np.copy(self.scale)
        #self.scale = 1
        #self.old_scale = 1




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
        fb_h = max(self.height() * dpr, 1.0)
        if self.mouse_down:
            self.screen_corner[0] = self.prev_screen_corner[0] - (event.position().x() * dpr - self.prev_mouse_coords[0])*self.scale/fb_h
            self.screen_corner[1] = self.prev_screen_corner[1] + (event.position().y() * dpr - self.prev_mouse_coords[1])*self.scale/fb_h

    def wheelEvent(self, event):
        print("Activating wheelEvent")
        print(type(self.L), self.L)
        print(type(self.scale), self.scale)
        delta = event.angleDelta().y()
        factor = 1.1

        dpr = self.devicePixelRatioF()
        fb_h = max(self.height() * dpr, 1.0)
        print("Old Scale: ", self.scale)
        print("Old Screen Corner: ", self.screen_corner)
        mouse_screen = np.array([event.position().x() * dpr/fb_h, 1 - event.position().y() * dpr/fb_h], dtype=np.float32)
        print("Mouse Screen: ", mouse_screen)
        mouse_pos = self.screen_corner + mouse_screen*self.scale
        print("Mouse Position: ", mouse_pos)
        self.old_scale = np.copy(self.scale)
        if delta > 0 and self.scale < 4*self.initial_scale:
            self.scale *= factor # smaller
            print("New Scale: ", self.scale)

        elif delta < 0 and self.scale > (1/4)*self.initial_scale:
            self.scale /= factor # bigger
            print("New Scale: ", self.scale)

        #self.scale = np.clip(self.scale, self.L * 0.01, self.L * 10)
        self.screen_corner = mouse_pos - mouse_screen*self.scale
        print("New Screen Corner: ", self.screen_corner)

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
        fb_h = max(self.height() * dpr, 1.0)
        fb_w = max(self.width() * dpr, 1.0)
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


        self.psi = self.psi*xp.exp(-1j*self.V*self.delta_t/self.hbar)
        psi_hat = xp.fft.fft2(self.psi)
        psi_hat = psi_hat*xp.exp(-1j*self.hbar*(self.k_0**2)*self.delta_t/(2*self.e_mass))
        self.psi = xp.fft.ifft2(psi_hat)

        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader)

        dpr = self.devicePixelRatioF()
        fb_h = max(self.height() * dpr, 1.0)
        fb_w = max(self.width() * dpr, 1.0)

        center_loc = glGetUniformLocation(self.shader, "center")
        glUniform2f(center_loc, self.center[0], self.center[1])

        screen_loc = glGetUniformLocation(self.shader, "screen")
        glUniform2f(screen_loc, self.screen_corner[0], self.screen_corner[1])

        scale_loc = glGetUniformLocation(self.shader, "scale")
        glUniform1f(scale_loc, self.scale)

        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform1f(res_loc, fb_h)

        max_loc = glGetUniformLocation(self.shader, "max")
        glUniform2f(max_loc, self.L, self.L*fb_h/fb_w)

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