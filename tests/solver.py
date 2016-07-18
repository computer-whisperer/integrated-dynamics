import int_dynamics.physics
import time
import numpy as np
import numpy.linalg
columns = []
for y in range(10):
    new_column = []
    for x in range(10):
        if x == y:
            new_column.append(1)
        elif y > 4 and x == 5:
            new_column.append(0.2)
        else:
            new_column.append(0)
    columns.append(new_column)
H = int_dynamics.physics.ExplicitMatrix(columns)
f = int_dynamics.physics.ExplicitMatrix([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
print("\n\nH:")
print(H.columns)
print("\n\nf:")
print(f.columns)
start_time = time.time()
result = H.solve(f)
end_time = time.time()
print("\n\nmy solution:")
print(result.columns)
print("\n solution took {} with ExplicitMatrix".format(end_time-start_time))
H_np = np.array(H.columns).T
f_np = np.array(f.columns).T
start_time = time.time()
result = numpy.linalg.solve(H_np, f_np)
end_time = time.time()
print("\n\nnumpy's solution:")
print(result)
print("\n solution took {} with numpy".format(end_time-start_time))
import theano
import theano.tensor as T
torque = T.scalar("torque")
f = int_dynamics.physics.ExplicitMatrix([[0, 0, 0, 0, 0, 0, torque, torque, torque, torque]])
result = H.solve(f)
fun = theano.function([torque], result.columns[0][-1])
start_time = time.time()
for i in range(0, 100):
    fun(i)
end_time = time.time()
print("\n\ntheano's solution with ExplicitMatrix:")
print(result)
print("\n solution took {} with theano".format((end_time-start_time)/100))
import theano.tensor.slinalg
H = T.dmatrix("H")
f = T.dmatrix("f")
result = theano.tensor.slinalg.solve(H, f)
fun = theano.function([H, f], result)
start_time = time.time()
res = fun(H_np, f_np)
end_time = time.time()
print("\n\ntheano's solution with scipy:")
print(res)
print("\n solution took {} with theano".format(end_time-start_time))