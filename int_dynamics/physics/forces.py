import sympy


class Force:

    def get_vector(self, body):
        return 0


class Gravity(Force):

    def get_vector(self, body):
        return -9.81*body.root_body.frame.y*body.body_mass


class Drag(Force):

    def __init__(self, coefficient):
        self.coefficient = coefficient

    def get_vector(self, body):
        return -body.point.vel(body.root_body.frame)


class Thruster(Force):

    def __init__(self, name, motor_controller, max_thrust):
        self.name = name
        self.motor_controller = motor_controller
        self.max_thrust = max_thrust

    def get_vector(self, body):
        return body.frame.y * self.motor_controller.get() * self.max_thrust
