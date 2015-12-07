import theano


class GearBox:
    """
    Simulates the dynamics of a gearbox with one or more motors attached
    """

    def __init__(self, motors, gear_ratio=10, dynamic_friction=10):
        self.motors = motors
        self.friction = dynamic_friction
        self.gear_ratio = gear_ratio
        self.state_functions = {}
        self.state = {
            "rotations": theano.shared(0.0)
        }

    def get_tensors(self, tensors_in):
        self.state_functions = {
            "rotations": theano.function([], self.state["rotations"]+tensors_in["rot_travel"]),
            "rps": theano.function([], tensors_in["rps"])
        }

        motor_rps = tensors_in["rps"]*self.gear_ratio

        torque_in = 0
        for motor in self.motors:
            motor_tensors = motor.get_tensors({"rps": motor_rps})["torque"]
            torque_in += motor_tensors
        torque_out = torque_in*self.gear_ratio

        return {
            "torque": torque_out
        }

    def update_state(self):
        rotations = self.state_functions["rotations"]()
        for motor in self.motors:
            motor.update_state()
        self.state["rotations"].set_value(rotations)
