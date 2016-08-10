import pickle
import time
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
import math

integrator = None
last_time = time.time()

def main():
    global integrator
    start_time = time.time()
    print("loading integrator from cache")
    with open("autocache.pkl", "rb") as f:
        integrator = pickle.load(f)
    print("load took {} seconds".format(time.time() - start_time))

    integrator.build_simulation_function()
    integrator.build_rendering_function()

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1000, 1000)
    glutCreateWindow(b'integrated_dynamics')

    glClearColor(0.,0.,0.,1.)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    lightZeroPosition = [10.,4.,10.,1.]
    lightZeroColor = [0.8,1.0,0.8,1.0] #green tinged
    glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
    glEnable(GL_LIGHT0)
    glutDisplayFunc(display)
    glutIdleFunc(display)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(40.,1.,1.,40.)
    glMatrixMode(GL_MODELVIEW)
    gluLookAt(0,0,10,
              0,0,0,
              0,1,0)
    glPushMatrix()
    glutMainLoop()

replay = 2

def display():
    global last_time
    current_time = time.time()
    dt = (current_time - last_time)/10
    if integrator.get_time() > replay:
        integrator.reset_simulation()
    integrator.step_time(min(dt, 0.1))
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glPushMatrix()
    color = [1.0,0.,0.,1.]
    glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
    integrator.opengl_draw()
    glPopMatrix()
    glutSwapBuffers()
    last_time = current_time

if __name__ == '__main__':
    main()
