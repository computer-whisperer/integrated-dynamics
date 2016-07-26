"""
This is a set of types designed to abstract the underlying math engine from integrated dynamics. The preferred engine
will always be theano, and this is in no way a replacement, but I needed to be able to evaluate these expressions without
the big theano dependency. This also seeks to pre-optimize the graph before theano performs its optimizations to reduce
function-building overhead.

The symbolic structure is largely a clone of how theano does it.
"""
import numpy as np
import math


class SymbolicConfig:
    backend = "numpy"
    do_optimize = True
    live_execute = True
    eval_id = 0


def get_value(node):
    if isinstance(node, Node):
        return node.get_value()
    else:
        return node


def count_nodes(node):
    node_list = []
    if isinstance(node, Node):
        node.enumerate_nodes(node_list)
        return len(node_list)
    else:
        return 0


class Node:

    parent_nodes = []
    last_eval_id = -1
    result = None

    def __init__(self):
        if SymbolicConfig.live_execute:
            self.get_value()

    def __add__(self, other):
        return AddOp.create_node([self, other])

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + other*-1

    def __rsub__(self, other):
        return self*-1 + other

    def __mul__(self, other):
        return MultOp.create_node([self, other])

    def __rmul__(self, other):
        return self*other

    def __truediv__(self, other):
        return self * InvertOp.create_node(other)

    def __neg__(self):
        return self*-1

    def __pow__(self, power, modulo=None):
        result = 0
        for _ in range(power):
            result *= power
        return result

    def get_value(self):
        raise NotImplementedError()

    def enumerate_nodes(self, node_list):
        if self not in node_list:
            node_list.append(self)
        for node in self.parent_nodes:
            if isinstance(node, Node):
                node.enumerate_nodes(node_list)

    def eval(self):
        return build_symbolic_function(self)()


class VariableNode(Node):

    def __init__(self, value):
        self.value = value
        self.value_changed = False
        self.shared = None
        super().__init__()

    def set_value(self, value):
        if value != self.value:
            self.value_changed = True
        self.value = value
        if self.shared is not None:
            self.shared.set_value(self.value)

    def get_value(self):
        if SymbolicConfig.backend == "theano":
            if self.shared is None:
                self.shared = SymbolicConfig.theano.shared(self.value, SymbolicConfig.theano.config.floatX)
            return self.shared
        else:
            if self.shared is not None:
                self.value = self.shared.get_value()
            return self.value

    def __getitem__(self, item):
        return SlicedNode(self, item)


class SlicedNode(Node):

    def __init__(self, node, item):
        self.node = node
        self.item = item
        super().__init__()

    def get_value(self):
        if self.last_eval_id < SymbolicConfig.eval_id:
            self.result = get_value(self.node)[self.item]
            self.last_eval_id = SymbolicConfig.eval_id
        return self.result

    def __getitem__(self, item):
        return SlicedNode(self, item)


class StackedNode(Node):

    def __init__(self, nodes, axis):
        self.parent_nodes = nodes
        self.axis = axis
        super().__init__()

    def get_value(self):
        if self.last_eval_id < SymbolicConfig.eval_id:
            vals = [get_value(node) for node in self.parent_nodes]
            if SymbolicConfig.backend == "theano":
                self.result = SymbolicConfig.T.stack(vals, axis=self.axis)
            else:
                self.result = np.stack(vals, axis=self.axis)
            self.last_eval_id = SymbolicConfig.eval_id
        return self.result


def stack(nodes, axis=0):
    if len(nodes) == 1:
        return nodes[0]
    return StackedNode(nodes, axis)


class ConcatNode(Node):

    def __init__(self, nodes, axis):
        self.parent_nodes = nodes
        self.axis = axis

    def get_value(self):
        if self.last_eval_id < SymbolicConfig.eval_id:
            vals = [get_value(node) for node in self.parent_nodes]
            if SymbolicConfig.backend == "theano":
                self.result = SymbolicConfig.T.concatenate(vals, axis=self.axis)
            else:
                self.result = np.concatenate(vals, axis=self.axis)
            self.last_eval_id = SymbolicConfig.eval_id
        return self.result


def concatenate(nodes, axis=0):
    if len(nodes) == 1:
        return nodes[0]
    return ConcatNode(nodes, axis)


class ApplyNode(Node):

    def __init__(self, op, inputs):
        self.op = op
        self.parent_nodes = inputs
        super().__init__()

    def get_value(self):
        if self.last_eval_id < SymbolicConfig.eval_id:
            self.result = self.op.execute([get_value(node) for node in self.parent_nodes])
            self.last_eval_id = SymbolicConfig.eval_id
        return self.result


class Op:

    @classmethod
    def create_node(cls, inputs):
        return ApplyNode(cls, inputs)

    @classmethod
    def execute(cls, inputs):
        raise NotImplementedError()


class AddOp(Op):

    @classmethod
    def create_node(cls, inputs):
        if not SymbolicConfig.do_optimize:
            return ApplyNode(cls, inputs)
        nodes, constant = cls.get_nodes_and_constant(inputs)
        if constant != 0:
            nodes.append(constant)
        if len(nodes) == 1:
            return nodes[0]
        return ApplyNode(cls, nodes)

    @classmethod
    def get_nodes_and_constant(cls, values):
        constant = 0
        nodes = []
        for value in values:
            if isinstance(value, Node):
                if isinstance(value, ApplyNode) and value.op is cls:
                    child_nodes, child_constant = cls.get_nodes_and_constant(value.parent_nodes)
                    constant += child_constant
                    nodes.extend(child_nodes)
                else:
                    nodes.append(value)
            else:
                constant += value
        return nodes, constant

    @classmethod
    def execute(cls, inputs):
        return sum(inputs)


class MultOp(Op):

    @classmethod
    def create_node(cls, inputs):
        if not SymbolicConfig.do_optimize:
            return ApplyNode(cls, inputs)
        nodes, constant = cls.get_nodes_and_constant(inputs)
        if constant == 0:
            return 0
        if constant != 1:
            nodes.append(constant)
        if len(nodes) == 1:
            return nodes[0]
        return ApplyNode(cls, nodes)

    @classmethod
    def get_nodes_and_constant(cls, values):
        constant = 1
        nodes = []
        for value in values:
            if isinstance(value, Node):
                if isinstance(value, ApplyNode) and value.op is cls:
                    child_nodes, child_constant = cls.get_nodes_and_constant(value.parent_nodes)
                    constant *= child_constant
                    nodes.extend(child_nodes)
                else:
                    nodes.append(value)
            else:
                constant *= value
        return nodes, constant

    @classmethod
    def execute(cls, inputs):
        result = 1
        for val in inputs:
            result *= val
        return result


class SqrtOp(Op):
    @classmethod
    def create_node(cls, input):
        if not SymbolicConfig.do_optimize:
            return ApplyNode(cls, [input])
        if isinstance(input, Node):
            return ApplyNode(cls, [input])
        else:
            return math.sqrt(input)

    @classmethod
    def execute(cls, inputs):
        if SymbolicConfig.backend == "theano":
            return SymbolicConfig.T.sqrt(inputs[0])
        else:
            return math.sqrt(inputs[0])

sqrt = SqrtOp.create_node


class InvertOp(Op):

    @classmethod
    def create_node(cls, value):
        if value == 0:
            raise ZeroDivisionError()
        return ApplyNode(cls, [value])

    @classmethod
    def execute(cls, inputs):
        return 1/inputs[0]


class SineOp(Op):
    @classmethod
    def create_node(cls, input):
        if not SymbolicConfig.do_optimize:
            return ApplyNode(cls, [input])
        if isinstance(input, Node):
            return ApplyNode(cls, [input])
        else:
            return math.sin(input)

    @classmethod
    def execute(cls, inputs):
        if SymbolicConfig.backend == "theano":
            return SymbolicConfig.T.sin(inputs[0])
        else:
            return math.sin(inputs[0])

sin = SineOp.create_node


class CosineOp(Op):
    @classmethod
    def create_node(cls, input):
        if not SymbolicConfig.do_optimize:
            return ApplyNode(cls, [input])
        if isinstance(input, Node):
            return ApplyNode(cls, [input])
        else:
            return math.cos(input)

    @classmethod
    def execute(cls, inputs):
        if SymbolicConfig.backend == "theano":
            return SymbolicConfig.T.cos(inputs[0])
        else:
            return math.sin(inputs[0])

cos = CosineOp.create_node

def _evaluate(node):
    SymbolicConfig.eval_id += 1
    return get_value(node)

def build_symbolic_function(node, backend=None):
    if backend is not None:
        SymbolicConfig.backend = backend
    if SymbolicConfig.backend == "theano":
        import theano
        import theano.tensor as T
        SymbolicConfig.theano = theano
        SymbolicConfig.T = T
        return theano.function([], _evaluate(node))
    else:
        return lambda: _evaluate(node)
