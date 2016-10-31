import dill
import time
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
import math
from int_dynamics.physics import *

integrator = None
last_time = time.time()

def main():
    global integrator
    start_time = time.time()
    #print("loading integrator from cache")
    #with open("autocache.pkl", "rb") as f:
    #    integrator = dill.load(f)
    #print("load took {} seconds".format(time.time() - start_time))
    world = WorldBody()
    link_1 = CubeBody(0.1, 1, 0.1, 1, name="link_1")
    link_1.forces.append((link_1.point, world.frame.y * (-9.81) * link_1.body_mass))
    # link_2 = CubeBody(0.5, 0.5, 0.1, 2, name="link_2")

    joint_1 = Joint.elbow_joint(
        joint_base_lin=[0, 2, 0],
        joint_motion_ang=[math.pi / 4, 0, 0],
        body_pose_lin=[0, -1, 0]
    )

    world.add_child(
        link_1,
        joint_1)
    #    pose=PoseVector(linear_component=XYVector(0, -0.75)),
    #    joint_base=PoseVector(linear_component=XYVector(0, 2), angular_component=Versor(XYZVector(0, 1, 0), 0)),
    #    joint_pose=PoseVector(angular_component=Versor(XYZVector(0, 0, 1), math.pi/3, symbols=True)),
    #    joint_motion=MotionVector(angular_component=XYZVector(0.75, 0, 0, symbols=True))
    # )

    # link_1.add_child(
    #    link_2,
    #    pose=PoseVector(linear_component=XYVector(0, -0.5)),
    #    joint_base=PoseVector(linear_component=XYVector(0, -0.5)),
    #    joint_pose=PoseVector(angular_component=Angle(0, symbols=False, use_constant=True)),
    #    joint_motion=MotionVector(angular_component=Angle(0, symbols=False, use_constant=False))
    # )

    integrator = EulerIntegrator("dual_pendulum")
    integrator.build_simulation_expressions(world, MotionVector(XYVector(0, 9.81), frame=world.frame))
    integrator.build_simulation_function()

    integrator.build_rendering_function()
    #integrator.current_state[8] = 0
    #integrator.current_state[13] = 10000
    #integrator.current_state[14] = 10000
    #integrator.current_state[15] = 10000
    #integrator.current_state[16] = 10000

    print(integrator.current_state)

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
    gluPerspective(40.,1.,1.,100.)
    glMatrixMode(GL_MODELVIEW)
    gluLookAt(0,0,10,
              0,0,0,
              0,1,0)
    glPushMatrix()
    glutMainLoop()

replay = 1000

def display():
    global last_time
    current_time = time.time()
    dt = (current_time - last_time)
    if integrator.get_time() > replay:
        integrator.reset_simulation()
    integrator.step_time(0.01)
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
