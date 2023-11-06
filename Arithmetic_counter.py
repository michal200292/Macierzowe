
class Counter:
    def __init__(self, add=0, mul=0, div=0):
        self.add = add
        self.mul = mul
        self.div = div

    def __add__(self, other):
        return Counter(self.add + other.add,
                       self.mul + other.mul,
                       self.div + other.div)

    def __repr__(self):
        return f"Counter({self.add}, {self.mul}, {self.div})"

    def get_params(self):
        return self.add, self.mul, self.div

