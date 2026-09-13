from math import sin
from calculator.advanced.root import square_root
# from ..advanced.root import square_root

def sinus(x:float) -> float:
    """
    Функция высчитывает синус угла.
    :param x: float
    :return: float
    """
    return sin(x)

if __name__ == '__main__':
    print(square_root(10))