from abc import ABC, abstractmethod


class MyAbstractClass(ABC):
    @abstractmethod
    def my_method(self, *args, **kwargs):
        """Abstract method, subclasses should implement this method with custom arguments."""

    @staticmethod
    @abstractmethod
    def create_spar(*args, **kwargs):
        """Creates a main spar choosing the optimal position for that type of spar.
        Usually by maximizing and objective function."""


class SubClassA(MyAbstractClass):
    def my_method(self, x, y):
        print(f"SubClassA: x={x}, y={y}")

    @staticmethod
    def create_spar(
        surface,
        thickness: float,
        p=None,
        n=None,
        chord_position=0.75,
    ):
        pass


class SubClassB(MyAbstractClass):
    def my_method(self, a, b, c=None):
        print(f"SubClassB: a={a}, b={b}, c={c}")

    @staticmethod
    def create_spar(
        surface,
        thickness: float,
        p,
        n,
        width: float,
        height: float,
        length: float,
    ):
        pass


class SubClassC(MyAbstractClass):
    def my_method(self, **kwargs):
        print(f"SubClassC: kwargs={kwargs}")

    @staticmethod
    def create_spar():
        pass


if __name__ == "__main__":
    a = SubClassA()
    b = SubClassB()
    c = SubClassC()
