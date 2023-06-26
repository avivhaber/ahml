from ahml.core import Scalar

a = Scalar(-4.0)
b = Scalar(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(g)
g.backward()
print(a.grad)
print(b.grad)
