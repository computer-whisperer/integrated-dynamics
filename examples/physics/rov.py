from int_dynamics.physics import *
import math

world = WorldBody()
rov = CubeBody(1, 1, 1, 1, name="ROV")

world.add_child(
    rov,
    Joint.free_joint())

gravity = Gravity()
drag = Drag(1)

depth = rov.point.pos_from(world.point)

rov.forces.append((rov.point, world.frame.y * -10 * depth.dot(world.frame.y)))
rov.add_force(Gravity())
rov.add_force(drag)

s2o2 = math.sqrt(2)/2

thruster_config = {
    "left_rear_thruster": {
        "pos": [-1, 0, -1],
        "ang": [s2o2, s2o2, 0, 0]
    },
    "right_rear_thruster": {
        "pos": [1, 0, -1],
        "ang": [s2o2, s2o2, 0, 0]
    },
    "left_front_thruster": {
        "pos": [-1, 0, 1],
        "ang": [s2o2, s2o2, 0, 0]
    },
    "right_front_thruster": {
        "pos": [1, 0, 1],
        "ang": [s2o2, s2o2, 0, 0]
    },
}
thruster_controllers = {}
for thruster in thruster_config:
    thruster_body = CubeBody(0.1, 0.5, 0.1, 0.2, name=thruster+"_body")
    thruster_controller = MotorController(name=thruster+"_controller")
    thruster_force = Thruster(thruster, thruster_controller, 10)
    joint = Joint.fixed_joint(thruster, thruster_config[thruster]["pos"], thruster_config[thruster]["ang"])
    rov.add_child(thruster_body, joint)
    thruster_body.add_force(thruster_force)
    thruster_controllers[thruster] = thruster_controller

integrator = EquationBuilder("rov")
integrator.add_controllers(thruster_controllers.values())
integrator.build_simulation_expressions(world)
integrator.build_simulation_function()
#integrator.current_state = [1.00,  -0.00, 0.00,  -0.09,  -0.00,  0.00,  -3.41]

renderer = OpenGLRenderer(integrator)
renderer.main_loop()



