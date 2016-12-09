from int_dynamics.physics import *
import math

world = WorldBody()
rov = CubeBody(1, 0.2, 1, 5, name="ROV")

world.add_child(
    rov,
    Joint.free_joint())

gravity = Gravity()
drag = Drag(1)
buoyancy = Buoyancy(500)

rov.add_force(gravity)
rov.add_force(drag)
rov.add_force(buoyancy)

s3o3 = math.sqrt(3)/3

a = 0.6532814824381883
b = 0.6532814824381882
c = 0.27059805007309845
d = 0.27059805007309845

thruster_config = {
    "left_rear_thruster": {
        "pos": [-0.5, 0.2, -0.5],
        "ang": [a, -b, -c, -d],
        "def_pow": 0.5
    },
    "right_rear_thruster": {
        "pos": [-0.5, 0.2, 0.5],
        "ang": [a, b, c, -d],
        "def_pow": 0.5
    },
    "left_front_thruster": {
        "pos": [0.5, 0.2, -0.5],
        "ang": [a, -b, c, d],
        "def_pow": -0.5
    },
    "right_front_thruster": {
        "pos": [0.5, 0.2, 0.5],
        "ang": [a, b, -c, d],
        "def_pow": -0.5
    },
    "left_vert_thruster": {
        "pos": [0, 0.2, -1],
        "ang": [1, 0, 0, 0],
        "def_pow": 0
    },
    "right_vert_thruster": {
        "pos": [0, 0.2, 1],
        "ang": [1, 0, 0, 0],
        "def_pow": 0.1
    },
}
thruster_controllers = {}
for thruster in thruster_config:
    thruster_body = CubeBody(0.1, 0.5, 0.1, 0.2, name=thruster+"_body")
    thruster_controller = MotorController(name=thruster+"_controller")
    thruster_controller.set(thruster_config[thruster]["def_pow"])
    thruster_force = ThrusterForce(thruster_controller, 10)
    joint = Joint.fixed_joint(thruster, thruster_config[thruster]["pos"], thruster_config[thruster]["ang"])
    rov.add_child(thruster_body, joint)
    thruster_body.add_force(thruster_force)
    thruster_body.add_force(gravity)
    thruster_body.add_force(drag)
    thruster_body.add_force(buoyancy)
    thruster_controllers[thruster] = thruster_controller

equation_builder = EquationBuilder("rov")
equation_builder.conts = thruster_controllers
equation_builder.add_controllers(thruster_controllers.values())
equation_builder.build_simulation_expressions(world, cse=True)
equation_builder.build_simulation_function("theano")
#integrator.current_state = [1.00,  -0.00, 0.00,  -0.09,  -0.00,  0.00,  -3.41]

renderer = OpenGLRenderer(equation_builder)
renderer.main_loop()



