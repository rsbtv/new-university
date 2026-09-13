from abc import ABC, abstractmethod

class Figure(ABC):
    def perimetr(self):
        return(sum(self.sides))

    @abstractmethod
    def area(self):
        pass

class Polygon(Figure):
    def __init__(self, sides):
        print(sides)
        if any(side <= 0 for side in sides):
            raise ValueError("Длины сторон должны быть положительными числами!")
        self.sides = sides

    def area(self):
        pass

class Square(Polygon):
    def __init__(self, side):
        super().__init__([side] * 4)
        self.side = side

    def area(self):
        return self.side ** 2

class Rectangle(Polygon):
    def __init__(self, width, height):
        super().__init__([width, height, width, height])
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Triangle(Polygon):
    def __init__(self, a, b, c):
        super().__init__([a, b, c])
        self.a = a
        self.b = b
        self.c = c

    def area(self):
        p = self.perimetr() / 2
        area = (p * (p - self.a) * (p - self.b) * (p - self.c)) ** 0.5
        return area

if __name__ == '__main__':
    a = Square(5)
    b = Rectangle(5, 4)
    c = Triangle(5, 7, 8)

    print(a.perimetr(), a.area())
    print(b.perimetr(), b.area())
    print(c.perimetr(), c.area())