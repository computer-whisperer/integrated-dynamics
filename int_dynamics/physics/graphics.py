from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import time


class OpenGLRenderer:
    last_time = 0
    replay = 30
    title_update_counter = 0

    def __init__(self, integrator):
        self.integrator = integrator
        integrator.build_rendering_function()

    def main_loop(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(1000, 1000)
        glutCreateWindow(b'integrated_dynamics')

        glClearColor(0., 0., 0., 1.)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        lightZeroPosition = [10., 4., 10., 1.]
        lightZeroColor = [0.8, 1.0, 0.8, 1.0]  # green tinged
        glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
        glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1)
        glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
        glEnable(GL_LIGHT0)
        glutDisplayFunc(self.display)
        glutIdleFunc(self.display)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(40., 1., 1., 100.)
        glMatrixMode(GL_MODELVIEW)
        gluLookAt(0, 0, 10,
                  0, 0, 0,
                  0, 1, 0)
        glPushMatrix()
        glutMainLoop()

    def display(self):
        func_start_time = time.time()

        dt = (func_start_time - self.last_time)
        if self.integrator.get_time() > self.replay:
            self.integrator.reset_simulation()

        start_time = func_start_time
        self.integrator.step_time(dt)
        current_time = time.time()
        simulation_time = current_time - start_time
        start_time = current_time

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        color = [1.0, 0., 0., 1.]
        glMaterialfv(GL_FRONT, GL_DIFFUSE, color)
        self.integrator.opengl_draw()
        glPopMatrix()
        glutSwapBuffers()

        current_time = time.time()
        render_time = current_time - start_time

        self.title_update_counter += 1
        if self.title_update_counter > 100:
            self.title_update_counter = 0
            glutSetWindowTitle("Integrated Dynamics [sim: {}, draw: {}]".format(simulation_time, render_time))

        self.last_time = func_start_time

