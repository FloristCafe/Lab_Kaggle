import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "v2"))

from equation_solver import EquationSolver


class EquationSolverTests(unittest.TestCase):
    def test_difference_resolves_both_tiles(self):
        solver = EquationSolver(3, 3)
        a, b = (0, 0), (0, 1)
        self.assertTrue(solver.add_equation([a, b], 1).independent)
        self.assertEqual(solver.feature_map()[a], 0.5)
        update = solver.add_equation([a], 0)
        self.assertEqual(update.newly_resolved, {a: 0, b: 1})
        self.assertEqual(solver.rank, 2)
        self.assertEqual(solver.feature_map()[a], 0)
        self.assertEqual(solver.feature_map()[b], 1)

    def test_redundant_equation_does_not_grow_pool(self):
        solver = EquationSolver(2, 2)
        cells = [(0, 0), (0, 1)]
        solver.add_equation(cells, 1)
        update = solver.add_equation(cells, 1)
        self.assertFalse(update.independent)
        self.assertEqual(update.redundant_cells, frozenset(cells))
        self.assertEqual(len(solver.equations), 1)
        self.assertEqual(solver.rank, 1)

    def test_binary_bounds_and_chain_resolution(self):
        solver = EquationSolver(3, 1)
        a, b, c = (0, 0), (1, 0), (2, 0)
        solver.add_equation([a, b], 1)
        solver.add_equation([b, c], 0)
        self.assertEqual(solver.values, {a: 1, b: 0, c: 0})

    def test_conflict_rejects_without_mutating_state(self):
        solver = EquationSolver(2, 1)
        cell = (0, 0)
        solver.add_equation([cell], 0)
        with self.assertRaises(ValueError):
            solver.add_equation([cell], 1)
        self.assertEqual(solver.values, {cell: 0})
        self.assertEqual(len(solver.equations), 1)

    def test_fractional_singleton_is_not_binary(self):
        solver = EquationSolver(3, 1)
        a, b, c = (0, 0), (1, 0), (2, 0)
        solver.add_equation([a, b], 1)
        solver.add_equation([b, c], 1)
        with self.assertRaises(ValueError):
            solver.add_equation([a, c], 1)
        self.assertEqual(solver.rank, 2)
        self.assertEqual(len(solver.equations), 2)


if __name__ == "__main__":
    unittest.main()
