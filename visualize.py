import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import QTimer, Qt
import sys
from classes import SimParams

params = SimParams("sim_params.npz")
frames = np.load("psi_vis_output.npy", "r")


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
        self.timer.start(16) # 16

        self.keys = set()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.mouse_down = False
        self.prev_mouse_coords = np.array([0, 0], dtype=np.float32)
        self.prev_screen_corner = np.array([0, 0], dtype=np.float32)


        self.scale = float(params.sim_dims[1])
        self.old_scale = float(params.sim_dims[1])
        self.initial_scale = np.copy(self.scale)
        self.frame_counter = 0



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

        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, params.W, params.H, 0, GL_RED, GL_FLOAT, None)

    def update_sim(self):
        self.frame_counter = 4000
        if self.frame_counter == params.max_frames:
            self.frame_counter = 0
        glBindTexture(GL_TEXTURE_2D, self.psi_texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, params.W, params.H, GL_RED, GL_FLOAT, frames[self.frame_counter])


        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader)

        dpr = self.devicePixelRatioF()
        fb_h = max(self.height() * dpr, 1.0)

        screen_loc = glGetUniformLocation(self.shader, "screen")
        glUniform2f(screen_loc, self.screen_corner[0], self.screen_corner[1])

        scale_loc = glGetUniformLocation(self.shader, "scale")
        glUniform1f(scale_loc, self.scale)

        res_loc = glGetUniformLocation(self.shader, "resolution")
        glUniform1f(res_loc, fb_h)

        max_loc = glGetUniformLocation(self.shader, "max")
        glUniform2f(max_loc, params.sim_dims[0], params.sim_dims[1])

        num_cell_loc = glGetUniformLocation(self.shader, "num_cells")
        glUniform2f(num_cell_loc, params.H, params.W)

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