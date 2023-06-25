class Scalar:
    @classmethod
    def wrap(cls, obj):
        return obj if isinstance(obj, Scalar) else Scalar(obj)

    def __init__(self, val, op=None, left=None, right=None):
        self.val = val
        self.op = op
        self.left = left
        self.right = right

    def __add__(self, other):
        other = Scalar.wrap(other)
        return Scalar(self.val + other.val, '+', self, other)
    
    def __mul__(self, other):
        other = Scalar.wrap(other)
        return Scalar(self.val * other.val, '*', self, other)
    
    def __pow__(self, other):
        other = Scalar.wrap(other)
        return Scalar(self.val ** other.val, '^', self, other)
    
    def __repr__(self):
        return str(self.val)