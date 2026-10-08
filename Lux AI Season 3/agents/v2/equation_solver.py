"""Binary coordinate constraints, independent of game observations and timing."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass(frozen=True)
class EquationUpdate:
    independent: bool
    newly_resolved: dict
    redundant_cells: frozenset


class EquationSolver:
    """Solve sums of binary tile values; unresolved tiles have a neutral 0.5 estimate."""

    def __init__(self, width, height):
        if width < 1 or height < 1:
            raise ValueError("map dimensions must be positive")
        self.width = width
        self.height = height
        self.equations = []
        self.dynamic_pool = []
        self.values = {}
        self.rank = 0
        self._basis = {}

    @property
    def static_pool(self):
        return self.equations

    def add_dynamic_equation(self, cells, score):
        group = frozenset(cells)
        self._validate(group, score)
        # A hidden relic can activate between observations. Never combine
        # dynamic equations or promote their conclusions to permanent values.
        self.dynamic_pool = [(group, score)]

    def enter_static_epoch(self):
        self.dynamic_pool.clear()

    def _validate(self, group, score):
        if any(not isinstance(cell, tuple) or len(cell) != 2
               or not all(isinstance(axis, int) for axis in cell)
               or not (0 <= cell[0] < self.width and 0 <= cell[1] < self.height)
               for cell in group):
            raise ValueError("equation contains an invalid map coordinate")
        if not isinstance(score, int) or not 0 <= score <= len(group):
            raise ValueError("score must be an integer between zero and cell count")

    @staticmethod
    def _reduce(rows, columns):
        matrix = [row.copy() for row in rows]
        pivots = []
        for column in range(columns):
            pivot = next((i for i in range(len(pivots), len(matrix))
                          if matrix[i][column]), None)
            if pivot is None:
                continue
            index = len(pivots)
            matrix[index], matrix[pivot] = matrix[pivot], matrix[index]
            divisor = matrix[index][column]
            matrix[index] = [value / divisor for value in matrix[index]]
            for i, row in enumerate(matrix):
                if i != index and row[column]:
                    factor = row[column]
                    matrix[i] = [a - factor * b for a, b in zip(row, matrix[index])]
            pivots.append(column)
        return matrix, len(pivots)

    @classmethod
    def _deduce(cls, equations):
        values = {}
        while True:
            cells = sorted({cell for group, _ in equations for cell in group
                            if cell not in values})
            rows = []
            for group, score in equations:
                remaining = score - sum(values[cell] for cell in group if cell in values)
                row = [Fraction(cell in group) for cell in cells]
                row.append(Fraction(remaining))
                rows.append(row)
            reduced, _ = cls._reduce(rows, len(cells))
            found = {}
            for row in reduced:
                coefficients, target = row[:-1], row[-1]
                nonzero = [value for value in coefficients if value]
                if len(nonzero) == 1 and target / nonzero[0] not in (0, 1):
                    raise ValueError("inconsistent binary equations")
                lower = sum(min(0, value) for value in coefficients)
                upper = sum(max(0, value) for value in coefficients)
                if target < lower or target > upper:
                    raise ValueError("inconsistent binary equations")
                if target == lower or target == upper:
                    for cell, coefficient in zip(cells, coefficients):
                        if coefficient:
                            found[cell] = int((coefficient > 0) == (target == upper))
            if not found:
                return values
            if any(cell in values and values[cell] != value
                   for cell, value in found.items()):
                raise ValueError("inconsistent binary equations")
            new = {cell: value for cell, value in found.items() if cell not in values}
            if not new:
                return values
            values.update(new)

    def add_equation(self, cells, score):
        group = frozenset(cells)
        self._validate(group, score)
        if (group, score) in self.equations:
            return EquationUpdate(False, {}, group)
        row = {cell: Fraction(1) for cell in group if cell not in self.values}
        rhs = Fraction(score - sum(self.values[cell] for cell in group
                                   if cell in self.values))
        for pivot in sorted(self._basis):
            coefficient = row.get(pivot, 0)
            if not coefficient:
                continue
            basis_row, basis_rhs = self._basis[pivot]
            for cell, value in basis_row.items():
                row[cell] = row.get(cell, 0) - coefficient * value
                if not row[cell]:
                    del row[cell]
            rhs -= coefficient * basis_rhs
        if not row:
            if rhs:
                raise ValueError("inconsistent binary equations")
            return EquationUpdate(False, {}, group)

        pivot = min(row)
        divisor = row[pivot]
        row = {cell: value / divisor for cell, value in row.items()}
        rhs /= divisor
        old_basis = {
            key: (value[0].copy(), value[1])
            for key, value in self._basis.items()
        }
        old_equations = self.equations.copy()
        old_values = self.values.copy()
        old_rank = self.rank
        self._basis[pivot] = (row, rhs)
        self.equations.append((group, score))
        self.rank += 1
        previous = self.values.copy()
        try:
            self._resolve_basis()
        except ValueError:
            self._basis = old_basis
            self.equations = old_equations
            self.values = old_values
            self.rank = old_rank
            raise
        newly_resolved = {cell: value for cell, value in self.values.items()
                          if cell not in previous}
        return EquationUpdate(True, newly_resolved, frozenset())

    def _resolve_basis(self):
        """Propagate binary bounds through the sparse incremental basis."""
        while True:
            found = {}
            for pivot, (basis_row, basis_rhs) in list(self._basis.items()):
                coefficients = {}
                rhs = basis_rhs
                for cell, coefficient in basis_row.items():
                    if cell in self.values:
                        rhs -= coefficient * self.values[cell]
                    else:
                        coefficients[cell] = coefficient
                self._basis[pivot] = (coefficients, rhs)
                if not coefficients:
                    if rhs:
                        raise ValueError("inconsistent binary equations")
                    continue
                lower = sum(min(Fraction(0), value)
                            for value in coefficients.values())
                upper = sum(max(Fraction(0), value)
                            for value in coefficients.values())
                if rhs < lower or rhs > upper:
                    raise ValueError("inconsistent binary equations")
                nonzero = list(coefficients.values())
                if len(nonzero) == 1 and rhs / nonzero[0] not in (0, 1):
                    raise ValueError("inconsistent binary equations")
                if rhs == lower or rhs == upper:
                    for cell, coefficient in coefficients.items():
                        found[cell] = int((coefficient > 0) == (rhs == upper))
            if any(cell in self.values and self.values[cell] != value
                   for cell, value in found.items()):
                raise ValueError("inconsistent binary equations")
            new = {cell: value for cell, value in found.items()
                   if cell not in self.values}
            if not new:
                return
            self.values.update(new)

    def feature_map(self):
        result = np.full((self.width, self.height), 0.5, dtype=np.float32)
        for cell, value in self.values.items():
            result[cell] = value
        return result
