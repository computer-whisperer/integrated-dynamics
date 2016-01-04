import theano

from examples.dynamics import get_dynamics

robot_dynamics_1 = get_dynamics()
results, _ = theano.scan(lambda: robot_dynamics_1.drivetrain.drivetrain_load.state_tensors["drivetrain"],
                         n_steps=100)
fn = theano.function([robot_dynamics_1.drivetrain.drivetrain_load.dt], [results])
res = fn(1)
print(res.size)
