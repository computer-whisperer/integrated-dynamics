from int_dynamics.physics import *


def test_basic_math():
    v1 = VariableNode(3)
    v2 = v1 + 5
    func = build_symbolic_function(v2, backend="not theano")
    assert func() == 8

    v1.set_value(2)
    assert func() == 7

    v3 = v1*v2
    func2 = build_symbolic_function(v3, backend="not theano")
    assert func2() == 14

    v4 = v3*0
    func3 = build_symbolic_function(v4, backend="not theano")
    assert func3() == 0

    v5 = v3 / v1
    func4 = build_symbolic_function(v5, backend="not theano")
    assert func4() == 7


def test_constant_optimizer():
    SymbolicConfig.do_optimize = False

    v1 = VariableNode(4)
    r1 = v1
    for i in range(100):
        r1 += i
    f1 = build_symbolic_function(r1, backend="not theano")
    assert f1() == 4954
    assert count_nodes(r1) > 100

    SymbolicConfig.do_optimize = True
    v2 = VariableNode(4)
    r2 = v2
    for i in range(100):
        r2 += i
    f2 = build_symbolic_function(r2, backend="not theano")
    assert f2() == 4954
    assert count_nodes(r2) < 10


def test_zero_optimizer():
    SymbolicConfig.do_optimize = False

    v1 = VariableNode(4)
    r1 = v1*0
    for i in range(100):
        r1 *= i
    f1 = build_symbolic_function(r1, backend="not theano")
    assert f1() == 0
    assert count_nodes(r1) > 100

    SymbolicConfig.do_optimize = True
    v2 = VariableNode(4)
    r2 = v2*0
    for i in range(100):
        r2 *= i
    f2 = build_symbolic_function(r2, backend="not theano")
    assert f2() == 0
    assert r2 == 0
