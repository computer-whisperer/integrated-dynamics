import theano


class GearBox:
    """
    Simulates the dynamics of a gearbox with one or more motors attached
    """

    def __init__(self, motors, gear_ratio=10, dynamic_friction=10):
        self.motors = motors
        self.friction = dynamic_friction
        self.gear_ratio = gear_ratio
        self.state_tensors = {}
        self.state = {
            "rotations": theano.shared(0.0)
        }

    def get_tensors(self, tensors_in):
        self.state_tensors = {
            "rotations": self.state["rotations"]+tensors_in["rot_travel"],
        }

        motor_rps = tensors_in["rps"]*self.gear_ratio

        torque_in = 0
        for motor in self.motors:
            torque_in += motor.get_tensors({"rps": motor_rps})["torque"]
        torque_out = torque_in*self.gear_ratio

        return {
            "torque": torque_out
        }

    def get_shared(self):
        return [(self.state["rotations"], self.state_tensors["rotations"])]
