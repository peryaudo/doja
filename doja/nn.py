class Parameter:
    def __init__(self, data):
        self.data = data

class Module:
    def __init__(self):
        self.parameters = []

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.parameters.append(value)
        super().__setattr__(name, value)