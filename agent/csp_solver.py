"""Generic constraint-satisfaction solver: node consistency, AC-3, backtracking.

This module knows nothing about wine. It is hand-written and modelled on
aima-python ``csp.py`` (Russell & Norvig, *Artificial Intelligence: A Modern
Approach*, chapter 6 "Constraint Satisfaction Problems"). Each docstring names
the AIMA function or textbook section it mirrors.

Differences from aima-python, all deliberate:

* Constraints are objects with an ``id`` so that every pruned value can be
  counted against the constraint that removed it (result monitoring).
* Two variables may be linked by several constraints.
* The search enumerates **all** solutions through a callback instead of
  returning the first one.
* Domains may hold thousands of values, so constraints expose index-friendly
  hooks (``has_support``, ``supported_values``, ``count_conflicts``) and forward
  checking swaps a domain for its surviving subset instead of copying domains
  at every node.
* Values are always tried in a canonical sorted order, so results are
  deterministic even though domains are sets.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from typing import Any

Assignment = dict[str, Any]
Arc = tuple[str, str]
#: One undo record of forward checking: ``(variable, domain it had before)``.
Removal = tuple[str, set]
#: ``value_key(csp, var, value)`` -> sortable key; smaller keys are tried first.
ValueKey = Callable[["CSP", str, Any], Any]


# --------------------------------------------------------------------------- #
# Options and statistics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SolverOptions:
    """Switches for the four techniques, so the UI can show what each one buys.

    The options change the statistics, never the set of solutions.
    """

    use_mrv: bool = True
    use_lcv: bool = True
    use_forward_checking: bool = True
    use_ac3: bool = True


@dataclass
class SearchStats:
    """Everything the solver measures; feeds ``Diagnostics`` in ``agent/models.py``.

    * ``pruned_by_constraint`` — constraint id -> values removed by node
      consistency and AC-3 (search-time pruning is in ``forward_check_prunes``).
    * ``ac3_revisions`` — calls to ``revise`` (AIMA ``revise``).
    * ``assignments`` — values tried during backtracking (``var = value``).
    * ``backtracks`` — values tried that failed at once or whose subtree held no
      solution.
    * ``variable_order`` — variables in the order the search first chose them.
    """

    initial_domain_sizes: dict[str, int] = field(default_factory=dict)
    pruned_by_constraint: dict[str, int] = field(default_factory=dict)
    domain_sizes_after_node_consistency: dict[str, int] = field(default_factory=dict)
    domain_sizes_after_ac3: dict[str, int] = field(default_factory=dict)
    ac3_revisions: int = 0
    assignments: int = 0
    backtracks: int = 0
    forward_check_prunes: int = 0
    solutions_found: int = 0
    variable_order: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #


class Constraint:
    """A named constraint over one or two variables.

    Mirrors the ``constraints(A, a, B, b)`` function of aima-python ``CSP``, but
    as an object so it can carry an ``id`` and index-backed shortcuts.

    Subclasses must implement :meth:`satisfied`. Binary subclasses over large
    domains should also override :meth:`has_support` (used by AC-3),
    :meth:`supported_values` (used by forward checking) and, if exact LCV is
    wanted, :meth:`count_conflicts`, replacing the default loops with index
    lookups.
    """

    def __init__(self, id: str, scope: Iterable[str]) -> None:  # noqa: A002
        self.id = id
        self.scope: tuple[str, ...] = tuple(scope)
        if len(self.scope) not in (1, 2):
            raise ValueError(f"constraint {id!r}: scope must name 1 or 2 variables")
        if len(set(self.scope)) != len(self.scope):
            raise ValueError(f"constraint {id!r}: scope variables must be distinct")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id!r}, {self.scope!r})"

    def other(self, var: str) -> str:
        """Return the scope variable that is not ``var`` (binary constraints only)."""
        a, b = self.scope
        if var == a:
            return b
        if var == b:
            return a
        raise KeyError(f"{var!r} is not in the scope of constraint {self.id!r}")

    def satisfied(self, assignment: Assignment) -> bool:
        """Return whether the constraint holds; every scope variable is assigned.

        AIMA: the ``constraints`` function of ``CSP``.
        """
        raise NotImplementedError

    def pair_ok(self, var: str, value: Any, other_value: Any) -> bool:
        """Return whether ``var=value`` is compatible with the other variable's value.

        Binary only. A thin wrapper over :meth:`satisfied`; subclasses may
        override it to skip building the assignment dict.
        """
        return self.satisfied({var: value, self.other(var): other_value})

    def has_support(self, var: str, value: Any, other_domain: Collection) -> bool:
        """Return whether some value in ``other_domain`` is compatible with ``var=value``.

        Binary only. This is the inner test of AIMA ``revise``. The default
        loops over ``other_domain``; override with an index lookup.
        """
        return any(self.pair_ok(var, value, other) for other in other_domain)

    def supported_values(self, var: str, value: Any, other_domain: Collection) -> set:
        """Return the subset of ``other_domain`` compatible with ``var=value``.

        Binary only. Used by :func:`forward_check` (AIMA ``forward_checking``).
        The result must be a subset of ``other_domain``. The default loops over
        ``other_domain``; override with an index intersection.
        """
        return {other for other in other_domain if self.pair_ok(var, value, other)}

    def count_conflicts(self, var: str, value: Any, other_domain: Collection) -> int:
        """Return how many values of ``other_domain`` ``var=value`` rules out.

        Binary only. Used by LCV (AIMA ``lcv`` / ``nconflicts``). The default
        derives it from :meth:`supported_values`.
        """
        return len(other_domain) - len(self.supported_values(var, value, other_domain))


class UnaryConstraint(Constraint):
    """A constraint on one variable: ``predicate(value)`` must be true."""

    def __init__(self, id: str, var: str, predicate: Callable[[Any], bool]) -> None:  # noqa: A002
        super().__init__(id, (var,))
        self.predicate = predicate

    def satisfied(self, assignment: Assignment) -> bool:
        """Return ``predicate(value)`` for the scope variable."""
        return bool(self.predicate(assignment[self.scope[0]]))


class BinaryConstraint(Constraint):
    """A constraint on two variables: ``predicate(value_a, value_b)`` must be true."""

    def __init__(
        self,
        id: str,  # noqa: A002
        a: str,
        b: str,
        predicate: Callable[[Any, Any], bool],
    ) -> None:
        super().__init__(id, (a, b))
        self.predicate = predicate

    def satisfied(self, assignment: Assignment) -> bool:
        """Return ``predicate(value_a, value_b)`` for the two scope variables."""
        a, b = self.scope
        return bool(self.predicate(assignment[a], assignment[b]))

    def pair_ok(self, var: str, value: Any, other_value: Any) -> bool:
        """Call the predicate directly, with the arguments in scope order."""
        if var == self.scope[0]:
            return bool(self.predicate(value, other_value))
        return bool(self.predicate(other_value, value))


# --------------------------------------------------------------------------- #
# The CSP
# --------------------------------------------------------------------------- #


class CSP:
    """A finite-domain CSP with unary and binary constraints.

    Mirrors the aima-python ``CSP`` class (AIMA section 6.1): ``variables``,
    ``domains`` and ``neighbors``. ``domains`` holds the *current* domains (what
    aima-python calls ``curr_domains``); the sets passed in are copied, so the
    caller's data is never modified.

    ``value_key`` is an optional cheap stand-in for LCV: when set and LCV is on,
    :func:`order_domain_values` sorts by ``value_key(csp, var, value)``
    (smallest first) instead of counting conflicts.
    """

    def __init__(
        self,
        domains: dict[str, Iterable],
        constraints: Iterable[Constraint],
        value_key: ValueKey | None = None,
    ) -> None:
        self.variables: list[str] = list(domains)
        self.domains: dict[str, set] = {var: set(values) for var, values in domains.items()}
        self.constraints: list[Constraint] = list(constraints)
        self.value_key = value_key

        self._on: dict[str, list[Constraint]] = {var: [] for var in self.variables}
        self._unary: dict[str, list[Constraint]] = {var: [] for var in self.variables}
        self._binary: dict[str, list[Constraint]] = {var: [] for var in self.variables}
        self._between: dict[Arc, list[Constraint]] = {}
        self.neighbors: dict[str, list[str]] = {var: [] for var in self.variables}

        for constraint in self.constraints:
            for var in constraint.scope:
                if var not in self.domains:
                    raise ValueError(f"constraint {constraint.id!r} names unknown variable {var!r}")
                self._on[var].append(constraint)
            if len(constraint.scope) == 1:
                self._unary[constraint.scope[0]].append(constraint)
                continue
            a, b = constraint.scope
            for x, y in ((a, b), (b, a)):
                self._binary[x].append(constraint)
                self._between.setdefault((x, y), []).append(constraint)
                if y not in self.neighbors[x]:
                    self.neighbors[x].append(y)

    def constraints_on(self, var: str) -> list[Constraint]:
        """Return every constraint whose scope includes ``var``, in the order given."""
        return self._on[var]

    def unary_constraints(self, var: str) -> list[Constraint]:
        """Return the unary constraints on ``var``, in the order given."""
        return self._unary[var]

    def binary_constraints(self, var: str) -> list[Constraint]:
        """Return the binary constraints involving ``var``, in the order given."""
        return self._binary[var]

    def constraints_between(self, a: str, b: str) -> list[Constraint]:
        """Return the binary constraints linking ``a`` and ``b`` (either scope order)."""
        return self._between.get((a, b), [])

    def domain_sizes(self) -> dict[str, int]:
        """Return the size of every current domain."""
        return {var: len(self.domains[var]) for var in self.variables}


def _sorted_values(values: Iterable) -> list:
    """Return ``values`` in a canonical order, so set iteration order never leaks out."""
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda v: (type(v).__name__, repr(v)))


def _count_prune(stats: SearchStats, constraint_id: str, n: int) -> None:
    stats.pruned_by_constraint[constraint_id] = stats.pruned_by_constraint.get(constraint_id, 0) + n


# --------------------------------------------------------------------------- #
# Task 4.2 — node consistency and AC-3
# --------------------------------------------------------------------------- #


def node_consistency(csp: CSP, stats: SearchStats) -> bool:
    """Make every variable node-consistent (AIMA section 6.2.1).

    Unary constraints are applied in the order they were given; a value is
    counted, in ``stats.pruned_by_constraint``, against the first constraint
    that removes it. Every unary constraint id gets an entry, even if it is 0.
    Returns ``False`` if some domain is left empty.
    """
    for constraint in csp.constraints:
        if len(constraint.scope) != 1:
            continue
        var = constraint.scope[0]
        domain = csp.domains[var]
        removed = [value for value in domain if not constraint.satisfied({var: value})]
        domain.difference_update(removed)
        _count_prune(stats, constraint.id, len(removed))
    return all(csp.domains[var] for var in csp.variables)


def revise(csp: CSP, xi: str, xj: str, stats: SearchStats) -> bool:
    """Remove the values of ``xi`` that have no support in ``xj``; AIMA ``revise``.

    A value goes when **any** constraint between the two variables has no
    support for it; it is counted against the first such constraint. Returns
    whether the domain of ``xi`` changed.
    """
    stats.ac3_revisions += 1
    constraints = csp.constraints_between(xi, xj)
    domain_i = csp.domains[xi]
    domain_j = csp.domains[xj]
    removed = []
    for value in domain_i:
        for constraint in constraints:
            if not constraint.has_support(xi, value, domain_j):
                removed.append(value)
                _count_prune(stats, constraint.id, 1)
                break
    domain_i.difference_update(removed)
    return bool(removed)


def ac3(csp: CSP, stats: SearchStats, queue: Iterable[Arc] | None = None) -> bool:
    """Make the CSP arc-consistent; AIMA ``AC3`` (figure 6.3).

    ``queue`` holds arcs ``(Xi, Xj)``; by default every arc, in variable
    declaration order. When ``revise`` shrinks ``Xi``, the arcs ``(Xk, Xi)`` for
    the other neighbours ``Xk`` are queued again (an arc already waiting is not
    queued twice). Returns ``False`` as soon as a domain is empty.
    """
    if not all(csp.domains[var] for var in csp.variables):
        return False
    if queue is None:
        queue = [(xi, xj) for xi in csp.variables for xj in csp.neighbors[xi]]
    pending: deque[Arc] = deque()
    waiting: set[Arc] = set()
    for arc in queue:
        if arc not in waiting:
            waiting.add(arc)
            pending.append(arc)

    while pending:
        arc = pending.popleft()
        waiting.discard(arc)
        xi, xj = arc
        if revise(csp, xi, xj, stats):
            if not csp.domains[xi]:
                return False
            for xk in csp.neighbors[xi]:
                back = (xk, xi)
                if xk != xj and back not in waiting:
                    waiting.add(back)
                    pending.append(back)
    return True


# --------------------------------------------------------------------------- #
# Task 4.3 — backtracking search with MRV, LCV and forward checking
# --------------------------------------------------------------------------- #


def select_unassigned_variable(csp: CSP, assignment: Assignment, use_mrv: bool = True) -> str:
    """Choose the next variable to assign; AIMA ``mrv`` / ``first_unassigned_variable``.

    With MRV: the smallest current domain; ties go to the variable with the
    most constraints on other unassigned variables (degree heuristic, AIMA
    section 6.3.1), then to the smaller name. Without MRV: declaration order.
    """
    unassigned = [var for var in csp.variables if var not in assignment]
    if not unassigned:
        raise ValueError("every variable is already assigned")
    if not use_mrv:
        return unassigned[0]

    def key(var: str) -> tuple[int, int, str]:
        degree = sum(1 for c in csp.binary_constraints(var) if c.other(var) not in assignment)
        return (len(csp.domains[var]), -degree, var)

    return min(unassigned, key=key)


def order_domain_values(
    csp: CSP, var: str, assignment: Assignment, use_lcv: bool = True
) -> list[Any]:
    """Return the current values of ``var`` in the order to try them; AIMA ``lcv``.

    With LCV: fewest values removed from unassigned neighbours' domains first
    (``Constraint.count_conflicts``), or, if the CSP has a ``value_key``, by
    that key. Ties, and the non-LCV order, follow the canonical sorted order of
    the values (AIMA ``unordered_domain_values``, made deterministic).
    """
    values = _sorted_values(csp.domains[var])
    if not use_lcv or len(values) < 2:
        return values
    if csp.value_key is not None:
        value_key = csp.value_key
        return sorted(values, key=lambda value: value_key(csp, var, value))

    watched = [
        (constraint, csp.domains[constraint.other(var)])
        for constraint in csp.binary_constraints(var)
        if constraint.other(var) not in assignment
    ]
    if not watched:
        return values
    return sorted(
        values,
        key=lambda value: sum(c.count_conflicts(var, value, domain) for c, domain in watched),
    )


def forward_check(
    csp: CSP,
    var: str,
    value: Any,
    assignment: Assignment,
    removals: list[Removal],
    stats: SearchStats,
) -> bool:
    """Prune unassigned neighbours of ``var`` given ``var=value``; AIMA ``forward_checking``.

    Returns ``False`` on a domain wipe-out. Every domain that shrinks is
    recorded in ``removals`` as ``(neighbour, previous domain)`` and replaced by
    its surviving subset, so nothing is copied and :func:`restore` undoes the
    pruning (also after a wipe-out) in time proportional to ``len(removals)``.
    """
    for constraint in csp.binary_constraints(var):
        other = constraint.other(var)
        if other in assignment:
            continue
        domain = csp.domains[other]
        kept = constraint.supported_values(var, value, domain)
        if len(kept) == len(domain):
            continue
        removals.append((other, domain))
        csp.domains[other] = kept if isinstance(kept, set) else set(kept)
        stats.forward_check_prunes += len(domain) - len(kept)
        if not kept:
            return False
    return True


def restore(csp: CSP, removals: list[Removal]) -> None:
    """Undo the pruning recorded in ``removals`` and empty the list; AIMA ``restore``."""
    while removals:
        var, domain = removals.pop()
        csp.domains[var] = domain


def _consistent(
    csp: CSP, var: str, value: Any, assignment: Assignment, check_assigned: bool
) -> bool:
    """Return whether ``var=value`` conflicts with nothing; AIMA ``nconflicts == 0``.

    Unary constraints are always checked, so the search is correct even on a
    CSP that was not made node-consistent. Constraints with already-assigned
    variables are checked when ``check_assigned`` is true, i.e. when forward
    checking is off and the domains have not been pruned against them.
    """
    for constraint in csp.unary_constraints(var):
        if not constraint.satisfied({var: value}):
            return False
    if check_assigned:
        for constraint in csp.binary_constraints(var):
            other = constraint.other(var)
            if other in assignment and not constraint.pair_ok(var, value, assignment[other]):
                return False
    return True


def backtracking_search(
    csp: CSP,
    options: SolverOptions | None = None,
    stats: SearchStats | None = None,
    on_solution: Callable[[Assignment], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> SearchStats:
    """Enumerate every solution by depth-first search; AIMA ``backtracking_search``.

    Unlike the textbook version (figure 6.5) this does not stop at the first
    solution: ``on_solution`` receives a copy of **every** complete consistent
    assignment, and the search carries on. ``should_stop`` is polled before
    each value is tried; once it returns true the search unwinds.

    Recursion depth equals the number of variables. With forward checking off,
    each value is checked against the already-assigned variables instead, so
    the solutions are the same under every option combination. The domains are
    left exactly as they were found. Returns ``stats``.
    """
    options = options or SolverOptions()
    stats = stats if stats is not None else SearchStats()
    use_fc = options.use_forward_checking
    n_variables = len(csp.variables)
    assignment: Assignment = {}
    seen = set(stats.variable_order)
    stopped = False

    def backtrack() -> bool:
        """Extend ``assignment``; return whether the subtree held a solution."""
        nonlocal stopped
        if len(assignment) == n_variables:
            stats.solutions_found += 1
            if on_solution is not None:
                on_solution(dict(assignment))
            return True

        var = select_unassigned_variable(csp, assignment, options.use_mrv)
        if var not in seen:
            seen.add(var)
            stats.variable_order.append(var)

        found_any = False
        for value in order_domain_values(csp, var, assignment, options.use_lcv):
            if should_stop is not None and should_stop():
                stopped = True
            if stopped:
                break
            stats.assignments += 1
            found = False
            if _consistent(csp, var, value, assignment, check_assigned=not use_fc):
                assignment[var] = value
                removals: list[Removal] = []
                if not use_fc or forward_check(csp, var, value, assignment, removals, stats):
                    found = backtrack()
                restore(csp, removals)
                del assignment[var]
            if found:
                found_any = True
            elif not stopped:
                stats.backtracks += 1
        return found_any

    backtrack()
    return stats


def solve_all(
    csp: CSP,
    options: SolverOptions | None = None,
    on_solution: Callable[[Assignment], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[Assignment], SearchStats]:
    """Find all solutions: node consistency -> AC-3 (if enabled) -> backtracking.

    Returns ``(solutions, stats)``. If ``on_solution`` is given, each solution
    is handed to it instead of being collected and the returned list is empty
    (for callers that keep only a bounded best-of heap). The propagation steps
    shrink ``csp.domains`` for good; the search itself leaves them unchanged.
    When propagation empties a domain the search is skipped.
    """
    options = options or SolverOptions()
    stats = SearchStats()
    stats.initial_domain_sizes = csp.domain_sizes()

    consistent = node_consistency(csp, stats)
    stats.domain_sizes_after_node_consistency = csp.domain_sizes()
    if consistent and options.use_ac3:
        consistent = ac3(csp, stats)
    stats.domain_sizes_after_ac3 = csp.domain_sizes()

    solutions: list[Assignment] = []
    if consistent:
        collect = on_solution if on_solution is not None else solutions.append
        backtracking_search(csp, options, stats, collect, should_stop)
    return solutions, stats
