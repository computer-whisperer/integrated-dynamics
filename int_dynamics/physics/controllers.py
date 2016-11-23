import sympy


class Controller:

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    def get_symbols(self):
        pass

    def get_values(self):
        pass


class MotorController(Controller):

    def __init__(self, name=None):
        super().__init__(name)
        self.symbol = sympy.symbols(self.name)
        self.value = 0

    def get_symbols(self):
        return [self.symbol]

    def get_values(self):
        return {self.symbol: self.value}

    def set(self, power):
        self.value = power

    def get(self):
        return self.symbol