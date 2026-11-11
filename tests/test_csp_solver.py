"""Tests for the generic CSP solver (plan tasks 4.1, 4.2 and 4.3)."""

from __future__ import annotations

import itertools
import random

import pytest

from agent.csp_solver import (
    CSP,
    BinaryConstraint,
    Constraint,
    SearchStats,
    SolverOptions,
    UnaryConstraint,
    ac3,
    backtracking_search,
    forward_check,
    node_consistency,
    order_domain_values,
    restore,
    revise,
    select_unassigned_variable,
    solve_all,
)

ALL_OPTIONS = [
    SolverOptions(use_mrv=mrv, use_lcv=lcv, use_forward_checking=fc, use_ac3=arc)
    for mrv, lcv, fc, arc in itertools.product([True, False], repeat=4)
]


def option_id(options: SolverOptions) -> str:
    flags = [
        ("mrv", options.use_mrv),
        ("lcv", options.use_lcv),
        ("fc", options.use_forward_checking),
        ("ac3", options.use_ac3),
    ]
    return "+".join(name for name, on in flags if on) or "plain"


def different(a: object, b: object) -> bool:
    return a != b


def queens_csp(n: int) -> CSP:
    """One variable per column, value = row; no two queens attack each other."""
    names = [f"Q{i}" for i in range(n)]
    constraints = [
        BinaryConstraint(
            f"Q{i}-Q{j}", names[i], names[j], lambda a, b, d=j - i: a != b and abs(a - b) != d
        )
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return CSP({name: set(range(n)) for name in names}, constraints)


AUSTRALIA_BORDERS = [
    ("WA", "NT"),
    ("WA", "SA"),
    ("NT", "SA"),
    ("NT", "Q"),
    ("SA", "Q"),
    ("SA", "NSW"),
    ("SA", "V"),
    ("Q", "NSW"),
    ("NSW", "V"),
]


def australia_csp() -> CSP:
    regions = ["WA", "NT", "SA", "Q", "NSW", "V", "T"]
    constraints = [BinaryConstraint(f"{a}!={b}", a, b, different) for a, b in AUSTRALIA_BORDERS]
    return CSP({r: {"red", "green", "blue"} for r in regions}, constraints)


def brute_force(domains: dict[str, set], constraints: list[Constraint]) -> set[tuple]:
    names = list(domains)
    found = set()
    for combo in itertools.product(*(sorted(domains[n]) for n in names)):
        assignment = dict(zip(names, combo, strict=True))
        if all(c.satisfied(assignment) for c in constraints):
            found.add(combo)
    return found


def as_set(solutions: list[dict], names: list[str]) -> set[tuple]:
    return {tuple(s[n] for n in names) for s in solutions}


# --------------------------------------------------------------------------- #
# Task 4.1 — representation
# --------------------------------------------------------------------------- #


def test_csp_structure():
    c_ab = BinaryConstraint("ab", "A", "B", different)
    c_ab2 = BinaryConstraint("ab2", "B", "A", lambda b, a: b > a)
    c_bc = BinaryConstraint("bc", "B", "C", different)
    u_a = UnaryConstraint("ua", "A", lambda v: v > 0)
    source = {"A": {1, 2}, "B": {1, 2}, "C": {1, 2}, "D": {1}}
    csp = CSP(source, [c_ab, u_a, c_ab2, c_bc])

    assert csp.variables == ["A", "B", "C", "D"]
    assert csp.neighbors == {"A": ["B"], "B": ["A", "C"], "C": ["B"], "D": []}
    assert csp.constraints_on("A") == [c_ab, u_a, c_ab2]
    assert csp.unary_constraints("A") == [u_a]
    assert csp.binary_constraints("B") == [c_ab, c_ab2, c_bc]
    assert csp.constraints_between("A", "B") == [c_ab, c_ab2]
    assert csp.constraints_between("B", "A") == [c_ab, c_ab2]
    assert csp.constraints_between("A", "C") == []
    assert csp.constraints_on("D") == []

    # The CSP owns copies of the domains.
    csp.domains["A"].discard(1)
    assert source["A"] == {1, 2}


def test_constraint_helpers():
    less = BinaryConstraint("lt", "X", "Y", lambda x, y: x < y)
    assert less.scope == ("X", "Y")
    assert less.other("X") == "Y" and less.other("Y") == "X"
    assert less.satisfied({"X": 1, "Y": 2, "Z": 99})
    assert not less.satisfied({"X": 2, "Y": 2})
    # Argument order follows the scope whichever side is asked about.
    assert less.pair_ok("X", 1, 2) and not less.pair_ok("Y", 1, 2)
    assert less.has_support("X", 2, {1, 2, 3})
    assert not less.has_support("X", 3, {1, 2, 3})
    assert less.has_support("Y", 2, {1, 2, 3})
    assert not less.has_support("Y", 1, {1, 2, 3})
    assert less.supported_values("X", 2, {1, 2, 3, 4}) == {3, 4}
    assert less.count_conflicts("X", 2, {1, 2, 3, 4}) == 2
    assert less.count_conflicts("Y", 2, {1, 2, 3, 4}) == 3

    positive = UnaryConstraint("pos", "X", lambda v: v > 0)
    assert positive.scope == ("X",)
    assert positive.satisfied({"X": 1}) and not positive.satisfied({"X": 0})


def test_invalid_constraints_are_rejected():
    with pytest.raises(ValueError):
        Constraint("three", ("A", "B", "C"))
    with pytest.raises(ValueError):
        Constraint("none", ())
    with pytest.raises(ValueError):
        BinaryConstraint("self", "A", "A", different)
    with pytest.raises(ValueError):
        CSP({"A": {1}}, [BinaryConstraint("ab", "A", "B", different)])
    with pytest.raises(NotImplementedError):
        Constraint("abstract", ("A",)).satisfied({"A": 1})


def test_base_class_defaults_work_from_satisfied_alone():
    class SumIsTen(Constraint):
        def satisfied(self, assignment):
            return assignment["X"] + assignment["Y"] == 10

    c = SumIsTen("sum10", ("X", "Y"))
    assert c.has_support("X", 4, {5, 6})
    assert not c.has_support("X", 1, {5, 6})
    assert c.supported_values("Y", 6, {3, 4, 5}) == {4}
    assert c.count_conflicts("Y", 6, {3, 4, 5}) == 2

    solutions, _ = solve_all(CSP({"X": set(range(10)), "Y": set(range(10))}, [c]))
    assert as_set(solutions, ["X", "Y"]) == {(x, 10 - x) for x in range(1, 10)}


# --------------------------------------------------------------------------- #
# Task 4.2 — node consistency
# --------------------------------------------------------------------------- #


def test_node_consistency_counts_against_first_removing_constraint():
    csp = CSP(
        {"X": set(range(1, 11)), "Y": {1, 2, 3}},
        [
            UnaryConstraint("even", "X", lambda v: v % 2 == 0),
            BinaryConstraint("x!=y", "X", "Y", different),
            UnaryConstraint("gt4", "X", lambda v: v > 4),
            UnaryConstraint("y_any", "Y", lambda v: True),
        ],
    )
    stats = SearchStats()
    assert node_consistency(csp, stats) is True
    assert csp.domains["X"] == {6, 8, 10}
    assert csp.domains["Y"] == {1, 2, 3}
    # "even" removes 1,3,5,7,9; "gt4" then only sees 2 and 4 (1 and 3 are already gone).
    assert stats.pruned_by_constraint == {"even": 5, "gt4": 2, "y_any": 0}


def test_node_consistency_reports_empty_domain():
    csp = CSP({"X": {1, 2}}, [UnaryConstraint("big", "X", lambda v: v > 5)])
    stats = SearchStats()
    assert node_consistency(csp, stats) is False
    assert csp.domains["X"] == set()
    assert stats.pruned_by_constraint == {"big": 2}


# --------------------------------------------------------------------------- #
# Task 4.2 — AC-3
# --------------------------------------------------------------------------- #


def three_region_csp(a: set, b: set, c: set) -> CSP:
    return CSP(
        {"A": a, "B": b, "C": c},
        [
            BinaryConstraint("A!=B", "A", "B", different),
            BinaryConstraint("A!=C", "A", "C", different),
            BinaryConstraint("B!=C", "B", "C", different),
        ],
    )


def test_ac3_map_colouring_full_domains_prunes_nothing():
    colours = {"red", "green", "blue"}
    csp = three_region_csp(colours, colours, colours)
    stats = SearchStats()
    assert ac3(csp, stats) is True
    assert all(csp.domains[v] == colours for v in "ABC")
    assert stats.pruned_by_constraint == {}
    assert stats.ac3_revisions == 6  # the six arcs, each revised once


def test_ac3_alone_solves_map_colouring_with_hand_worked_counts():
    csp = three_region_csp({"red"}, {"red", "green"}, {"red", "green", "blue"})
    stats = SearchStats()
    assert ac3(csp, stats) is True
    assert csp.domains == {"A": {"red"}, "B": {"green"}, "C": {"blue"}}
    # Queue: (A,B) (A,C) (B,A) (B,C) (C,A) (C,B).
    #   (B,A) drops red from B   -> A!=B: 1; (C,B) is already waiting.
    #   (C,A) drops red from C   -> A!=C: 1; re-queues (B,C).
    #   (C,B) drops green from C -> B!=C: 1; re-queues (A,C).
    # 6 initial revisions + 2 re-queued = 8.
    assert stats.pruned_by_constraint == {"A!=B": 1, "A!=C": 1, "B!=C": 1}
    assert stats.ac3_revisions == 8

    # With every domain a singleton the search has nothing left to decide.
    search = backtracking_search(csp, SolverOptions(), SearchStats())
    assert search.solutions_found == 1
    assert search.assignments == 3
    assert search.backtracks == 0


def test_ac3_textbook_square_example():
    # AIMA section 6.2.2: Y = X^2 over the digits.
    digits = set(range(10))
    csp = CSP(
        {"X": digits, "Y": digits},
        [BinaryConstraint("y=x^2", "X", "Y", lambda x, y: y == x * x)],
    )
    stats = SearchStats()
    assert ac3(csp, stats) is True
    assert csp.domains == {"X": {0, 1, 2, 3}, "Y": {0, 1, 4, 9}}
    assert stats.pruned_by_constraint == {"y=x^2": 12}  # 6 from X, 6 from Y
    assert stats.ac3_revisions == 2


def test_ac3_propagates_along_a_chain():
    # X < Y < Z over {1,2,3}: only X=1, Y=2, Z=3 survives.
    d = {1, 2, 3}
    csp = CSP(
        {"X": d, "Y": d, "Z": d},
        [
            BinaryConstraint("x<y", "X", "Y", lambda x, y: x < y),
            BinaryConstraint("y<z", "Y", "Z", lambda y, z: y < z),
        ],
    )
    stats = SearchStats()
    assert ac3(csp, stats) is True
    assert csp.domains == {"X": {1}, "Y": {2}, "Z": {3}}
    # Queue: (X,Y) (Y,X) (Y,Z) (Z,Y).
    #   (X,Y) drops 3 from X          x<y: 1
    #   (Y,X) drops 1 from Y          x<y: 2   re-queues nothing new ((Z,Y) waiting)
    #   (Y,Z) drops 3 from Y          y<z: 1   re-queues (X,Y)
    #   (Z,Y) drops 1,2 from Z        y<z: 3
    #   (X,Y) drops 2 from X          x<y: 3
    assert stats.pruned_by_constraint == {"x<y": 3, "y<z": 3}
    assert stats.ac3_revisions == 5


def test_ac3_detects_inconsistency():
    d = {1, 2, 3}
    csp = CSP(
        {"X": d, "Y": d, "Z": d},
        [
            BinaryConstraint("x<y", "X", "Y", lambda x, y: x < y),
            BinaryConstraint("y<z", "Y", "Z", lambda y, z: y < z),
            BinaryConstraint("z<x", "Z", "X", lambda z, x: z < x),
        ],
    )
    stats = SearchStats()
    assert ac3(csp, stats) is False
    assert any(not csp.domains[v] for v in csp.variables)

    solutions, solve_stats = solve_all(
        CSP(
            {"X": d, "Y": d, "Z": d},
            csp.constraints,
        )
    )
    assert solutions == []
    assert solve_stats.assignments == 0  # AC-3 proved it; no search needed
    assert 0 in solve_stats.domain_sizes_after_ac3.values()


def test_ac3_counts_against_first_unsupporting_constraint_between_a_pair():
    csp = CSP(
        {"X": {1, 2, 3, 4}, "Y": {2, 3}},
        [
            BinaryConstraint("x<=y", "X", "Y", lambda x, y: x <= y),
            BinaryConstraint("x!=y-1", "X", "Y", lambda x, y: x != y - 1),
        ],
    )
    stats = SearchStats()
    assert revise(csp, "X", "Y", stats) is True
    # 4 has no y with x<=y. 1, 2, 3 each have support under both constraints
    # taken separately (1: y=3; 2: y=2; 3: y=3).
    assert csp.domains["X"] == {1, 2, 3}
    assert stats.pruned_by_constraint == {"x<=y": 1}
    assert stats.ac3_revisions == 1
    assert revise(csp, "X", "Y", stats) is False
    assert stats.ac3_revisions == 2


def test_ac3_accepts_an_explicit_queue():
    d = {1, 2, 3}
    csp = CSP({"X": d, "Y": d}, [BinaryConstraint("x<y", "X", "Y", lambda x, y: x < y)])
    stats = SearchStats()
    assert ac3(csp, stats, queue=[("X", "Y")]) is True
    assert csp.domains == {"X": {1, 2}, "Y": {1, 2, 3}}  # only the X side was revised
    assert stats.ac3_revisions == 1


def test_ac3_uses_has_support_override():
    class IndexedEquals(Constraint):
        """X == Y, answered by a set lookup instead of a loop."""

        def __init__(self):
            super().__init__("eq", ("X", "Y"))
            self.lookups = 0

        def satisfied(self, assignment):
            raise AssertionError("AC-3 must go through has_support")

        def has_support(self, var, value, other_domain):
            self.lookups += 1
            return value in other_domain

    constraint = IndexedEquals()
    csp = CSP({"X": set(range(0, 1000)), "Y": set(range(500, 1500))}, [constraint])
    stats = SearchStats()
    assert ac3(csp, stats) is True
    assert csp.domains["X"] == csp.domains["Y"] == set(range(500, 1000))
    assert stats.pruned_by_constraint == {"eq": 1000}
    assert constraint.lookups == 2000  # one lookup per value, not per pair


# --------------------------------------------------------------------------- #
# Task 4.3 — variable and value ordering
# --------------------------------------------------------------------------- #


def test_select_unassigned_variable_mrv_degree_and_name():
    csp = CSP(
        {"D": {1, 2, 3}, "C": {1, 2}, "B": {1, 2}, "A": {1, 2}},
        [
            BinaryConstraint("cd", "C", "D", different),
            BinaryConstraint("bd", "B", "D", different),
            BinaryConstraint("ab", "A", "B", different),
        ],
    )
    # Without MRV: declaration order.
    assert select_unassigned_variable(csp, {}, use_mrv=False) == "D"
    assert select_unassigned_variable(csp, {"D": 1}, use_mrv=False) == "C"
    # MRV: A, B, C tie on size 2; B has degree 2, A and C degree 1.
    assert select_unassigned_variable(csp, {}, use_mrv=True) == "B"
    # With B assigned, A has degree 0 and C degree 1.
    assert select_unassigned_variable(csp, {"B": 1}, use_mrv=True) == "C"
    # With B and D assigned, A and C tie on size and degree: name decides.
    assert select_unassigned_variable(csp, {"B": 1, "D": 2}, use_mrv=True) == "A"
    # Smallest domain beats degree.
    csp.domains["D"] = {3}
    assert select_unassigned_variable(csp, {}, use_mrv=True) == "D"
    with pytest.raises(ValueError):
        select_unassigned_variable(csp, {"A": 1, "B": 2, "C": 1, "D": 3})


def test_order_domain_values_lcv():
    csp = CSP(
        {"X": {1, 2, 3}, "Y": {1, 2, 3}, "Z": {1, 2}},
        [
            BinaryConstraint("x<y", "X", "Y", lambda x, y: x < y),
            BinaryConstraint("x!=z", "X", "Z", different),
        ],
    )
    # Without LCV: canonical sorted order.
    assert order_domain_values(csp, "X", {}, use_lcv=False) == [1, 2, 3]
    # Conflicts: X=1 -> 1 in Y + 1 in Z = 2; X=2 -> 2 + 1 = 3; X=3 -> 3 + 0 = 3.
    assert order_domain_values(csp, "X", {}, use_lcv=True) == [1, 2, 3]
    # Y: Y=3 rules out 1 value of X (3), Y=2 two, Y=1 three.
    assert order_domain_values(csp, "Y", {}, use_lcv=True) == [3, 2, 1]
    # Assigned neighbours do not count: with Y assigned only Z matters for X.
    assert order_domain_values(csp, "X", {"Y": 3}, use_lcv=True) == [3, 1, 2]


def test_order_domain_values_uses_count_conflicts_hook():
    class Scripted(BinaryConstraint):
        def count_conflicts(self, var, value, other_domain):
            return {"a": 5, "b": 0, "c": 2}[value]

    csp = CSP(
        {"X": {"a", "b", "c"}, "Y": {1, 2}},
        [Scripted("s", "X", "Y", lambda x, y: True)],
    )
    assert order_domain_values(csp, "X", {}, use_lcv=True) == ["b", "c", "a"]
    assert order_domain_values(csp, "X", {}, use_lcv=False) == ["a", "b", "c"]


def test_order_domain_values_uses_value_key():
    calls = []

    def value_key(csp, var, value):
        calls.append((var, value))
        return -value

    csp = CSP(
        {"X": {1, 2, 3}, "Y": {1, 2, 3}},
        [BinaryConstraint("x<y", "X", "Y", lambda x, y: x < y)],
        value_key=value_key,
    )
    assert order_domain_values(csp, "X", {}, use_lcv=True) == [3, 2, 1]
    assert sorted(calls) == [("X", 1), ("X", 2), ("X", 3)]
    # value_key is an LCV substitute: it is ignored when LCV is off.
    assert order_domain_values(csp, "X", {}, use_lcv=False) == [1, 2, 3]


def test_values_of_mixed_types_still_get_a_stable_order():
    csp = CSP({"X": {1, "a", (2, 3)}}, [])
    first = order_domain_values(csp, "X", {}, use_lcv=False)
    assert sorted(map(repr, first)) == sorted(map(repr, [1, "a", (2, 3)]))
    assert order_domain_values(csp, "X", {}, use_lcv=False) == first


# --------------------------------------------------------------------------- #
# Task 4.3 — forward checking
# --------------------------------------------------------------------------- #


def test_forward_check_prunes_and_restores():
    csp = queens_csp(4)
    before = {var: set(domain) for var, domain in csp.domains.items()}
    stats = SearchStats()
    removals = []

    assert forward_check(csp, "Q0", 0, {"Q0": 0}, removals, stats) is True
    assert csp.domains["Q0"] == {0, 1, 2, 3}  # the assigned variable is untouched
    assert csp.domains["Q1"] == {2, 3}  # loses row 0 and the diagonal row 1
    assert csp.domains["Q2"] == {1, 3}
    assert csp.domains["Q3"] == {1, 2}
    assert stats.forward_check_prunes == 6
    assert [var for var, _ in removals] == ["Q1", "Q2", "Q3"]

    restore(csp, removals)
    assert removals == []
    assert csp.domains == before


def test_forward_check_skips_assigned_neighbours():
    csp = queens_csp(4)
    stats = SearchStats()
    removals = []
    assignment = {"Q0": 0, "Q1": 2}
    assert forward_check(csp, "Q1", 2, assignment, removals, stats) is True
    assert csp.domains["Q0"] == {0, 1, 2, 3}
    assert csp.domains["Q2"] == {0}  # rows 1, 2, 3 attacked by Q1=2
    assert csp.domains["Q3"] == {1, 3}  # rows 2 and 0 attacked
    assert stats.forward_check_prunes == 5


def test_forward_check_wipeout_can_be_restored():
    csp = CSP(
        {"X": {1, 2}, "Y": {1, 2, 3}, "Z": {1}},
        [
            BinaryConstraint("x<y", "X", "Y", lambda x, y: x < y),
            BinaryConstraint("x!=z", "X", "Z", different),
        ],
    )
    before = {var: set(domain) for var, domain in csp.domains.items()}
    stats = SearchStats()
    removals = []
    assert forward_check(csp, "X", 1, {"X": 1}, removals, stats) is False
    assert csp.domains["Z"] == set()
    assert stats.forward_check_prunes == 2  # 1 from Y, 1 from Z
    restore(csp, removals)
    assert csp.domains == before


def test_forward_check_uses_supported_values_override():
    class IndexedEquals(Constraint):
        def __init__(self):
            super().__init__("eq", ("X", "Y"))

        def satisfied(self, assignment):
            return assignment["X"] == assignment["Y"]

        def pair_ok(self, var, value, other_value):
            raise AssertionError("forward checking must go through supported_values")

        def supported_values(self, var, value, other_domain):
            return {value} & other_domain

    csp = CSP({"X": set(range(100)), "Y": set(range(50, 150))}, [IndexedEquals()])
    options = SolverOptions(use_ac3=False)
    solutions, stats = solve_all(csp, options)
    assert as_set(solutions, ["X", "Y"]) == {(v, v) for v in range(50, 100)}
    assert csp.domains == {"X": set(range(100)), "Y": set(range(50, 150))}
    assert stats.forward_check_prunes > 0


# --------------------------------------------------------------------------- #
# Task 4.3 — backtracking search
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("options", ALL_OPTIONS, ids=option_id)
@pytest.mark.parametrize(("n", "expected"), [(4, 2), (6, 4)])
def test_n_queens_solution_counts_under_all_flag_combinations(n, expected, options):
    csp = queens_csp(n)
    solutions, stats = solve_all(csp, options)
    assert len(solutions) == expected
    assert stats.solutions_found == expected
    names = csp.variables
    assert as_set(solutions, names) == brute_force(
        {name: set(range(n)) for name in names}, csp.constraints
    )
    assert len(as_set(solutions, names)) == expected  # no duplicates
    assert sorted(stats.variable_order) == sorted(names)


def test_four_queens_exact_solutions():
    solutions, _ = solve_all(queens_csp(4))
    assert as_set(solutions, ["Q0", "Q1", "Q2", "Q3"]) == {(1, 3, 0, 2), (2, 0, 3, 1)}


def test_eight_queens_has_92_solutions():
    solutions, stats = solve_all(queens_csp(8))
    assert len(solutions) == 92
    assert stats.solutions_found == 92


@pytest.mark.parametrize("options", ALL_OPTIONS, ids=option_id)
def test_australia_three_colours_has_18_solutions(options):
    csp = australia_csp()
    solutions, stats = solve_all(csp, options)
    assert len(solutions) == 18
    assert len(as_set(solutions, csp.variables)) == 18
    for solution in solutions:
        assert set(solution) == set(csp.variables)
        for a, b in AUSTRALIA_BORDERS:
            assert solution[a] != solution[b]
    assert stats.initial_domain_sizes == {v: 3 for v in csp.variables}


def test_forward_checking_never_needs_more_assignments_on_six_queens():
    for mrv, lcv, arc in itertools.product([True, False], repeat=3):
        with_fc = SolverOptions(mrv, lcv, True, arc)
        without_fc = SolverOptions(mrv, lcv, False, arc)
        _, stats_fc = solve_all(queens_csp(6), with_fc)
        _, stats_plain = solve_all(queens_csp(6), without_fc)
        assert stats_fc.assignments <= stats_plain.assignments
        assert stats_fc.forward_check_prunes > 0
        assert stats_plain.forward_check_prunes == 0
    # In plain declaration order the saving is strict.
    _, stats_fc = solve_all(queens_csp(6), SolverOptions(False, False, True, False))
    _, stats_plain = solve_all(queens_csp(6), SolverOptions(False, False, False, False))
    assert stats_fc.assignments < stats_plain.assignments


@pytest.mark.parametrize("options", ALL_OPTIONS, ids=option_id)
def test_domains_are_fully_restored_after_search(options):
    csp = queens_csp(6)
    before = {var: set(domain) for var, domain in csp.domains.items()}
    identities = {var: id(domain) for var, domain in csp.domains.items()}
    found = []
    stats = backtracking_search(csp, options, SearchStats(), found.append)
    assert len(found) == 4
    assert stats.solutions_found == 4
    assert csp.domains == before
    assert {var: id(domain) for var, domain in csp.domains.items()} == identities


def test_solve_all_leaves_domains_as_propagation_left_them():
    csp = three_region_csp({"red"}, {"red", "green"}, {"red", "green", "blue"})
    solutions, stats = solve_all(csp)
    assert solutions == [{"A": "red", "B": "green", "C": "blue"}]
    assert csp.domains == {"A": {"red"}, "B": {"green"}, "C": {"blue"}}
    assert stats.initial_domain_sizes == {"A": 1, "B": 2, "C": 3}
    assert stats.domain_sizes_after_node_consistency == {"A": 1, "B": 2, "C": 3}
    assert stats.domain_sizes_after_ac3 == {"A": 1, "B": 1, "C": 1}


@pytest.mark.parametrize("options", ALL_OPTIONS, ids=option_id)
def test_unsatisfiable_csp_returns_no_solutions(options):
    # Three mutually different variables, two colours: arc-consistent but unsolvable.
    two = {"red", "green"}
    csp = three_region_csp(two, two, two)
    solutions, stats = solve_all(csp, options)
    assert solutions == []
    assert stats.solutions_found == 0
    assert stats.assignments > 0
    assert stats.backtracks > 0
    assert csp.domains == {"A": two, "B": two, "C": two}


@pytest.mark.parametrize("options", ALL_OPTIONS, ids=option_id)
def test_unsatisfiable_by_unary_constraint_skips_search(options):
    csp = CSP(
        {"X": {1, 2}, "Y": {1, 2}},
        [
            UnaryConstraint("big", "X", lambda v: v > 5),
            BinaryConstraint("x!=y", "X", "Y", different),
        ],
    )
    solutions, stats = solve_all(csp, options)
    assert solutions == []
    assert stats.pruned_by_constraint == {"big": 2}
    assert stats.domain_sizes_after_node_consistency == {"X": 0, "Y": 2}
    assert stats.domain_sizes_after_ac3 == {"X": 0, "Y": 2}
    assert stats.assignments == 0
    assert stats.ac3_revisions == 0


def test_backtrack_and_assignment_counts_hand_worked():
    # X != Y over {1, 2}, plain search in declaration order:
    #   X=1: Y=1 fails (backtrack 1), Y=2 solution
    #   X=2: Y=1 solution, Y=2 fails (backtrack 2)
    csp = CSP({"X": {1, 2}, "Y": {1, 2}}, [BinaryConstraint("x!=y", "X", "Y", different)])
    solutions, stats = solve_all(csp, SolverOptions(False, False, False, False))
    assert solutions == [{"X": 1, "Y": 2}, {"X": 2, "Y": 1}]
    assert stats.assignments == 6
    assert stats.backtracks == 2
    assert stats.variable_order == ["X", "Y"]

    # With forward checking the failing Y values are pruned, never tried.
    csp = CSP({"X": {1, 2}, "Y": {1, 2}}, [BinaryConstraint("x!=y", "X", "Y", different)])
    solutions, stats = solve_all(csp, SolverOptions(False, False, True, False))
    assert solutions == [{"X": 1, "Y": 2}, {"X": 2, "Y": 1}]
    assert stats.assignments == 4
    assert stats.backtracks == 0
    assert stats.forward_check_prunes == 2


def test_variable_order_follows_mrv():
    csp = CSP(
        {"big": {1, 2, 3, 4}, "mid": {1, 2, 3}, "small": {1, 2}},
        [],
    )
    _, stats = solve_all(csp)
    assert stats.variable_order == ["small", "mid", "big"]
    assert stats.solutions_found == 24
    _, stats = solve_all(csp, SolverOptions(use_mrv=False))
    assert stats.variable_order == ["big", "mid", "small"]


def test_backtracking_search_checks_unary_constraints_itself():
    # Called directly, without node consistency, the search must still be right.
    for options in ALL_OPTIONS:
        csp = CSP(
            {"X": {1, 2, 3}, "Y": {1, 2, 3}},
            [
                UnaryConstraint("odd", "X", lambda v: v % 2 == 1),
                BinaryConstraint("x<y", "X", "Y", lambda x, y: x < y),
            ],
        )
        found = []
        backtracking_search(csp, options, SearchStats(), found.append)
        assert as_set(found, ["X", "Y"]) == {(1, 2), (1, 3)}


def test_on_solution_gets_independent_copies():
    found = []
    solve_all(australia_csp(), on_solution=found.append)
    assert len(found) == 18
    assert len({id(s) for s in found}) == 18
    assert len(as_set(found, list(found[0]))) == 18


def test_solve_all_with_callback_returns_empty_list():
    seen = []
    solutions, stats = solve_all(queens_csp(6), on_solution=seen.append)
    assert solutions == []
    assert len(seen) == 4
    assert stats.solutions_found == 4


@pytest.mark.parametrize("options", ALL_OPTIONS, ids=option_id)
def test_should_stop_ends_the_search_early(options):
    solutions, stats = solve_all(australia_csp(), options, should_stop=lambda: True)
    assert solutions == []
    assert stats.assignments == 0
    assert stats.backtracks == 0  # an interrupted search proves nothing

    solutions_seen: list[dict] = []
    csp = australia_csp()
    before = {var: set(domain) for var, domain in csp.domains.items()}
    _, stats = solve_all(
        csp,
        options,
        on_solution=solutions_seen.append,
        should_stop=lambda: len(solutions_seen) >= 5,
    )
    assert len(solutions_seen) == 5
    assert stats.solutions_found == 5
    assert csp.domains == before  # restored even when the search is cut short


def test_search_is_deterministic():
    first, stats_a = solve_all(queens_csp(6))
    second, stats_b = solve_all(queens_csp(6))
    assert first == second  # same solutions in the same order
    assert stats_a == stats_b


def test_csp_without_variables_has_one_empty_solution():
    solutions, stats = solve_all(CSP({}, []))
    assert solutions == [{}]
    assert stats.solutions_found == 1


def test_recursion_depth_equals_number_of_variables():
    # 300 variables in a chain of equalities: one frame per variable, well below
    # Python's default recursion limit, however many values are tried.
    n = 300
    names = [f"v{i:03d}" for i in range(n)]
    constraints = [
        BinaryConstraint(f"eq{i}", names[i], names[i + 1], lambda a, b: a == b)
        for i in range(n - 1)
    ]
    solutions, stats = solve_all(CSP({name: {0, 1} for name in names}, constraints))
    assert len(solutions) == 2
    assert stats.variable_order[0] == names[0]


def random_csp(seed: int) -> tuple[dict[str, set], list[Constraint]]:
    """A small random CSP with unary constraints and parallel binary constraints."""
    rng = random.Random(seed)
    names = [f"V{i}" for i in range(rng.randint(3, 5))]
    domains = {name: set(range(rng.randint(2, 5))) for name in names}
    constraints: list[Constraint] = []
    for index, name in enumerate(names):
        if rng.random() < 0.4:
            banned = rng.choice(sorted(domains[name]))
            constraints.append(UnaryConstraint(f"u{index}", name, lambda v, b=banned: v != b))
    for i, j in itertools.combinations(range(len(names)), 2):
        for copy in range(2):
            if rng.random() < 0.45:
                allowed = frozenset(
                    (a, b)
                    for a in domains[names[i]]
                    for b in domains[names[j]]
                    if rng.random() < 0.6
                )
                scope = (names[i], names[j]) if rng.random() < 0.5 else (names[j], names[i])
                if scope[0] == names[i]:
                    table = allowed
                else:
                    table = frozenset((b, a) for a, b in allowed)
                constraints.append(
                    BinaryConstraint(
                        f"b{i}{j}{copy}", scope[0], scope[1], lambda x, y, t=table: (x, y) in t
                    )
                )
    return domains, constraints


@pytest.mark.parametrize("seed", range(40))
def test_random_csps_match_brute_force_under_all_flag_combinations(seed):
    domains, constraints = random_csp(seed)
    expected = brute_force(domains, constraints)
    names = list(domains)
    for options in ALL_OPTIONS:
        csp = CSP(domains, constraints)
        solutions, stats = solve_all(csp, options)
        assert len(solutions) == len(expected), option_id(options)
        assert as_set(solutions, names) == expected, option_id(options)
        assert stats.solutions_found == len(expected)
        assert stats.initial_domain_sizes == {n: len(domains[n]) for n in names}
