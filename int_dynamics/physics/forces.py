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


class ThrusterForce(Force):

    def __init__(self, motor_controller, max_thrust):
        self.motor_controller = motor_controller
        self.max_thrust = max_thrust

    def get_vector(self, body):
        return body.frame.y * self.motor_controller.get() * self.max_thrust


class Buoyancy(Force):

    def __init__(self, coefficient):
        self.coefficient = coefficient

    def get_vector(self, body):
        altitude = body.point.pos_from(body.root_body.point).dot(body.root_body.frame.y)
        return body.root_body.frame.y * -self.coefficient * altitude * body.get_volume() / body.body_mass