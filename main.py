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

uniform vec2 screen;
uniform vec2 max;
uniform float resolution;
uniform float scale;
uniform vec2 num_cells;
uniform sampler2D psi_tex;

void main() {
    vec2 uv = gl_FragCoord.xy * scale / resolution + screen;
    float padding = 5e-10;
    if (uv.x > 0 && uv.x < max.x && uv.y > 0 && uv.y < max.y) {
        float x_cell = uv.x / max.x;
        float y_cell = uv.y / max.y;
        vec2 tex_uv = vec2(x_cell, y_cell);
        float vis = texture(psi_tex, tex_uv).r;
        if (vis < 0.33) {
            FragColor = vec4(3*vis, 0, 0, 1);
        }
        else if (vis > 0.66) {
            FragColor = vec4(1, 1, 3*(vis - 0.66), 1);
        }
        else {
            FragColor = vec4(1, 3*(vis - 0.33), 0, 1);
        }
    }
    else if (uv.x > -padding && uv.x < max.x + padding && uv.y > -padding && uv.y < max.y + padding) {
        FragColor = vec4(0.5, 0.5, 0.5, 1);
    }
    else {
        FragColor = vec4(0, 0, 0, 1);
    }
}
"""


class GLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.screen_corner = np.array([0, 0], dtype=np.float32)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_sim)
        self.timer.start(16)

        self.keys = set()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.mouse_down = False
        self.prev_mouse_coords = np.array([0, 0], dtype=np.float32)
        self.prev_screen_corner = np.array([0, 0], dtype=np.float32)
        self.sim_ratio = 0.57


        self.e_mass = 9.109e-31
        c = 3e8
        e_vel_x = 0.01*c
        #e_vel_x = 0
        e_vel_y = 0
        self.hbar = 1.05457e-34
        k_0_x = e_vel_x*self.e_mass/self.hbar
        k_0_y = e_vel_y*self.e_mass/self.hbar
        self.k_0 = xp.hypot(k_0_x, k_0_y)
        e_wavelength = 2*xp.pi/self.k_0
        sigma = 2e-9
        self.cell_spacing = sigma/(20*2)
        self.L = 12*sigma
        self.delta_t = self.e_mass*self.cell_spacing**2/(1*self.hbar) # 8
        self.sim_dims = [5*self.L, 5*self.L*self.sim_ratio]
        x_i = self.sim_dims[0]/4
        #x_i = self.sim_dims[0]/2
        y_i = self.sim_dims[1]/2

        self.num_cells = [int(self.sim_dims[0]/self.cell_spacing), int(self.sim_dims[1]/self.cell_spacing)]

        width_cells = xp.linspace(0, self.sim_dims[0], self.num_cells[0])
        height_cells = xp.linspace(0, self.sim_dims[1], self.num_cells[1])
        A, B = xp.meshgrid(width_cells, height_cells)
        cell_pos = xp.stack([A, B], axis=-1)

        # Initial Values
        self.psi = xp.exp((1j)*(k_0_x*(cell_pos[..., 0] - x_i) + k_0_y*(cell_pos[..., 1] - y_i)) - ((cell_pos[..., 0] - x_i)**2 + (cell_pos[..., 1] - y_i)**2)/(0.2*sigma**2))

        # Integrate
        psi_prob_int = (xp.abs(self.psi)**2)
        psi_prob_int = xp.sum(psi_prob_int)
        psi_prob_int *= self.cell_spacing**2

        # Normalize wavefunction
        self.psi /= xp.sqrt(psi_prob_int)

        V_real = xp.zeros_like(self.psi)

        x = xp.arange(self.num_cells[0])
        y = xp.arange(self.num_cells[1])

        X, Y = xp.meshgrid(x, y)
        mask = ((self.num_cells[0]/2 - 40 < X) & (X < self.num_cells[0]/2 + 40))
        mask &= ((Y < self.num_cells[1]/2 - 120) | (Y > self.num_cells[1]/2 + 120) | ((Y < self.num_cells[1]/2 + 40) & (Y > self.num_cells[1]/2 - 40)))
        V_real[mask] = 1e50

        dx = xp.minimum(x, self.num_cells[0] - x - 1)
        dy = xp.minimum(y, self.num_cells[1] - y - 1)

        DX, DY = xp.meshgrid(dx, dy)
        dist = xp.minimum(DX, DY)
        width = 3.5*sigma/ self.cell_spacing
        mask = xp.clip((width - dist)/width, 0, 1)
        E = 0.5 * self.e_mass * (e_vel_x**2 + e_vel_y**2)
        W = 1.5*E*mask**4

        self.V = V_real - 1j*W

        self.scale = float(self.sim_dims[1])
        self.old_scale = float(self.sim_dims[1])
        self.initial_scale = np.copy(self.scale)



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
        delta = event.angleDelta().y()
        factor = 1.1

        dpr = self.devicePixelRatioF()
        fb_h = max(self.height() * dpr, 1.0)
        mouse_screen = np.array([event.position().x() * dpr/fb_h, 1 - event.position().y() * dpr/fb_h], dtype=np.float32)
        mouse_pos = self.screen_corner + mouse_screen*self.scale
        self.old_scale = np.copy(self.scale)
        if delta > 0 and self.scale < 4*self.initial_scale:
            self.scale *= factor # smaller

        elif delta < 0 and self.scale > (1/4)*self.initial_scale:
            self.scale /= factor # bigger

        self.screen_corner = mouse_pos - mouse_screen*self.scale

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

        self.psi_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.psi_texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, self.psi.shape[1], self.psi.shape[0], 0, GL_RED, GL_FLOAT, None)

    def update_sim(self):

        self.psi = self.psi*xp.exp(-1j*self.V*self.delta_t/self.hbar)
        kx = xp.fft.fftfreq(int(self.sim_dims[0]/self.cell_spacing), d=self.cell_spacing)*2*xp.pi
        ky = xp.fft.fftfreq(int(self.sim_dims[1]/self.cell_spacing), d=self.cell_spacing)*2*xp.pi
        KX, KY = xp.meshgrid(kx, ky)
        k_squared = KX**2 + KY**2
        psi_hat = xp.fft.fft2(self.psi)
        psi_hat = psi_hat*xp.exp(-1j*self.hbar*k_squared*self.delta_t/(2*self.e_mass))
        self.psi = xp.fft.ifft2(psi_hat)
        psi_prob = xp.square(xp.abs(self.psi))
        psi_vis = xp.log1p(psi_prob)

        if not hasattr(self, "max_vis"):
            self.max_vis = xp.max(psi_vis)

        psi_vis /= self.max_vis
        psi_vis = xp.power(psi_vis, 0.4)
        psi_vis = xp.clip(psi_vis, 0, 1.0)
        psi_vis_cpu = xp.asnumpy(psi_vis)

        glBindTexture(GL_TEXTURE_2D, self.psi_texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, psi_vis_cpu.shape[1], psi_vis_cpu.shape[0], GL_RED, GL_FLOAT, psi_vis_cpu)

        #print("Max: ", np.max(psi_vis_cpu))
        #print("Min: ", np.min(psi_vis_cpu), "\n")

        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader)

        dpr = self.devicePixelRatioF()
        fb_h = max(self.height() * dpr, 1.0)
        fb_w = max(self.width() * dpr, 1.0)

        screen_loc = glGetUniformLocation(self.shader, "screen")
        glUniform2f(screen_loc, self.screen_corner[0], self.screen_corner[1])

        scale_loc = glGetUniformLocation(self.shader, "scale")
        glUniform1f(scale_loc, self.scale)

        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform1f(res_loc, fb_h)

        max_loc = glGetUniformLocation(self.shader, "max")
        glUniform2f(max_loc, self.sim_dims[0], self.sim_dims[1])

        num_cell_loc = glGetUniformLocation(self.shader, "num_cells")
        glUniform2f(num_cell_loc, self.num_cells[0], self.num_cells[1])

        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.psi_texture)

        tex_loc = glGetUniformLocation(self.shader, "psi_tex")
        glUniform1i(tex_loc, 0)



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