from collections.abc import Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class AdamsBashforth:
    """
    Adams-Bashforth methods for solving Ordinary Differential Equations (ODEs).

    Attributes:
        func: An ODE function of x and y.
        x_initials: List containing initial required values of x.
        y_initials: List containing initial required values of y.
        step_size: The increment value of x.
        x_final: The final value of x.
    
    Example usage:
        >>> def f(x, y):
        ...     return x + y
        >>> solver = AdamsBashforth(f, [0, 0.2, 0.4], [0, 0.2, 1], 0.2, 1)
        >>> solver.step_3()  # doctest: +ELLIPSIS
        array([0.   , 0.2  , 1.   , ...])
    """
    func: Callable[[float, float], float]
    x_initials: list[float]
    y_initials: list[float]
    step_size: float
    x_final: float

    def __post_init__(self) -> None:
        if self.x_initials[-1] >= self.x_final:
            raise ValueError(
                "The final value of x must be greater than the initial values of x."
            )

        if self.step_size <= 0:
            raise ValueError("Step size must be positive.")

        if not all(
            round(x1 - x0, 10) == self.step_size
            for x0, x1 in zip(self.x_initials, self.x_initials[1:])
        ):
            raise ValueError("x-values must be equally spaced according to step size.")

    def _validate_initial_points(self, required_points: int) -> None:
        if len(self.x_initials) != required_points or len(self.y_initials) != required_points:
            raise ValueError(f"Insufficient initial points information. Required: {required_points}")

    def step_2(self) -> np.ndarray:
        self._validate_initial_points(2)
        return self._adams_bashforth(2)

    def step_3(self) -> np.ndarray:
        self._validate_initial_points(3)
        return self._adams_bashforth(3)

    def step_4(self) -> np.ndarray:
        self._validate_initial_points(4)
        return self._adams_bashforth(4)

    def step_5(self) -> np.ndarray:
        self._validate_initial_points(5)
        return self._adams_bashforth(5)

    def _adams_bashforth(self, order: int) -> np.ndarray:
        coeffs = {
            2: [3, -1],
            3: [23, -16, 5],
            4: [55, -59, 37, -9],
            5: [1901, -2774, 2616, -1274, 251]
        }[order]

        x_vals = np.array(self.x_initials[:order])
        y_vals = np.array(self.y_initials[:order])
        n = int((self.x_final - x_vals[-1]) / self.step_size)
        y = np.zeros(n + order)
        y[:order] = y_vals

        for i in range(n):
            f_vals = np.array([self.func(x, y[j]) for j, x in enumerate(x_vals)])
            y[order + i] = y[order + i - 1] + (self.step_size / np.math.factorial(order - 1)) * np.dot(coeffs, f_vals[::-1])
            x_vals = np.append(x_vals[1:], x_vals[-1] + self.step_size)

        return y


if __name__ == "__main__":
    import doctest
    doctest.testmod()
