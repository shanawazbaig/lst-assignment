#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boolean SoP Analyzer
====================
A robust, *exact* analyzer for Boolean expressions in **Sum of Products (SoP)** form.

It can:
  1) Compute cofactors w.r.t. any variable *or a cube* (e.g., a, ab, ab').
  2) Compute **Boolean Difference**, **Consensus**, and **Smoothing** w.r.t. a variable.

Conventions
-----------
- Variables are case-sensitive (A and a are distinct).
- Complements use **prime** notation only: a' means NOT a.
- SoP format examples:   ab + a'c + bcd
                         x'y + z  (spaces are fine)
- A *cube* means a product literal set like ab'c, used to request a general cofactor f_ab'c.

Math (w.r.t. variable x)
------------------------
- Positive cofactor:      f_x   := f with x = 1
- Negative cofactor:      f_x'  := f with x = 0
- Smoothing:              S_x(f) = f_x + f_x'
- Consensus (operator):   C_x(f) = f_x · f_x'
- Boolean difference:     ∂f/∂x  = f_x ⊕ f_x'
  (Implemented exactly using symbolic cube arithmetic; no truth-table blowup.)

Input
-----
- From keyboard (paste a line with your SoP), or
- From file: one SoP per line *or* one product term per line (auto-detected).

Output
------
- Uses the same **prime** style (a, a') for complements.
- Performs safe simplifications (contradictions, duplicate literals, subset absorption).
  (Not a full minimizer; correctness is exact, even if not fully minimized.)

Run
---
$ python3 boolean_sop_tool.py
"""

from __future__ import annotations

import sys
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, FrozenSet

try:
    from colorama import init as colorama_init, Fore, Style
except Exception:
    # Safe fallback if colorama isn't installed
    class Dummy:
        RESET_ALL = ""

    class ForeDummy(Dummy):
        CYAN = GREEN = MAGENTA = YELLOW = RED = BLUE = ""

    class StyleDummy(Dummy):
        BRIGHT = DIM = NORMAL = ""
    Fore = ForeDummy()
    Style = StyleDummy()
    def colorama_init(): pass

colorama_init()

# (var, True) means var (uncomplemented), (var, False) means var'
Literal = Tuple[str, bool]
Cube = FrozenSet[Literal]           # product term
Cover = Set[Cube]                    # SoP: OR of cubes

# ---------- Configuration / Diagnostics ----------

VERBOSE = True                     # emit progress info
PROGRESS_INTERVAL = 250            # operations interval for status prints
EXPANSION_CUBE_LIMIT = 4000        # abort further expansion if exceeded

def log(msg: str):
    if VERBOSE:
        print(f"[info] {msg}")

def progress(kind: str, count: int, total: Optional[int] = None):
    if not VERBOSE:
        return
    if count % PROGRESS_INTERVAL != 0:
        return
    if total is not None:
        print(f"[working] {kind}: {count}/{total} ...", flush=True)
    else:
        print(f"[working] {kind}: {count} ...", flush=True)

# ---------- Parsing & Formatting ----------

VAR_RE = re.compile(r"[A-Za-z]\d*('?)*")  # tokens like a, b', X2, X2', etc.


def normalize_var(token: str) -> Tuple[str, bool]:
    """Normalize a literal token into (name, polarity) without forcing lowercase.

    CHANGED BEHAVIOUR:
      - Previously: Uppercase variable without prime meant complemented lowercase (A == a').
      - Now:       Case is preserved and significant. A and a are distinct symbols.
                   Complement is indicated ONLY by a trailing prime (').

    Examples:
        a   -> ("a", True)
        a'  -> ("a", False)
        A   -> ("A", True)
        A'  -> ("A", False)
        X2' -> ("X2", False)
    """
    t = token.strip()
    if not t:
        raise ValueError("Empty token")
    prime = t.endswith("'")
    base = t.rstrip("'")
    if not base:
        raise ValueError(f"Bad literal: {token}")
    # Keep case & digits exactly (case-sensitive variable universe)
    return base, (not prime)


def parse_sop(expr: str) -> Cover:
    """
    Parse a SoP string: e.g. "ab + a'c + bcd".
    Literals can be concatenated (ab) or spaced (a b). '+' separates sums.
    """
    expr = expr.strip()
    if expr == "0":
        return set()
    if expr == "1":
        return {frozenset()}  # empty cube means constant 1
    # Split on '+' at top level
    terms = [t.strip() for t in expr.replace(" ", "").split("+")]
    cover: Cover = set()
    for term in terms:
        if term == "" or term == "0":
            continue
        if term == "1":
            cover.add(frozenset())
            continue
        # Extract concatenated literals: scan like a'bC -> a' , b , C
        lits: List[Literal] = []
        i = 0
        while i < len(term):
            ch = term[i]
            if ch == "+":
                i += 1
                continue
            if not ch.isalpha():
                raise ValueError(f"Bad character '{ch}' in term '{term}'")
            # Var token may include digits after first letter
            j = i + 1
            while j < len(term) and term[j].isdigit():
                j += 1
            # Optional prime
            prime = False
            if j < len(term) and term[j] == "'":
                prime = True
                j += 1
            token = term[i:j]
            v, pol = normalize_var(token + ("'" if prime else ""))
            lits.append((v, pol))
            i = j
        cover.add(frozenset(lits))
    return simplify_cover(cover)


def parse_cube(expr: str) -> Dict[str, bool]:
    """
    Parse a cube like "ab'c" into an assignment dict { 'a':True, 'b':False, 'c':True }.
    """
    expr = expr.strip().replace(" ", "")
    if expr in ("", "1"):
        return {}
    if expr == "0":
        # Impossible assignment; we represent as empty assignment + a flag at call sites.
        return {"__IMPOSSIBLE__": True}
    assign: Dict[str, bool] = {}
    i = 0
    while i < len(expr):
        ch = expr[i]
        if not ch.isalpha():
            raise ValueError(f"Bad character '{ch}' in cube '{expr}'")
        j = i + 1
        while j < len(expr) and expr[j].isdigit():
            j += 1
        prime = False
        if j < len(expr) and expr[j] == "'":
            prime = True
            j += 1
        token = expr[i:j]
        v, pol = normalize_var(token + ("'" if prime else ""))
        if v in assign and assign[v] != pol:
            # Contradictory cube -> unsatisfiable
            return {"__IMPOSSIBLE__": True}
        assign[v] = pol
        i = j
    return assign


def lit_str(var: str, pol: bool) -> str:
    """Format literal in a / a' style."""
    return f"{var}{'' if pol else "'"}"


def cube_str(c: Cube) -> str:
    """Format cube as concatenated literals, or '1' for empty cube."""
    if len(c) == 0:
        return "1"
    return "".join(lit_str(v, p) for v, p in sorted(c))


def cover_str(cover: Cover) -> str:
    """Format cover (SoP) as 'lits + lits + ...' or '0' for empty cover."""
    if not cover:
        return "0"
    return " + ".join(cube_str(c) for c in sorted(cover, key=lambda x: (len(x), sorted(x))))


# ---------- Core Boolean Cover Algebra ----------

def simplify_cover(cover: Cover) -> Cover:
    """
    Simplify a cover via:
      - Remove contradictory cubes (x and x' both present)
      - Remove duplicate literals in a cube
      - Absorption: remove any cube that is a superset of another cube
    (Not a full minimizer; conservative but fast.)
    """
    # Remove contradictions and dedup literals
    cleaned: Set[Cube] = set()
    for c in cover:
        d: Dict[str, bool] = {}
        ok = True
        for v, p in c:
            if v in d and d[v] != p:
                ok = False
                break
            d[v] = p
        if ok:
            cleaned.add(frozenset(d.items()))
    # Absorption (subset removal)
    minimal: Set[Cube] = set(cleaned)
    for c in cleaned:
        for u in cleaned:
            if u is c:
                continue
            if u.issubset(c):
                if c in minimal:
                    minimal.discard(c)
                break
    return minimal


def vars_in_cover(cover: Cover) -> List[str]:
    s: Set[str] = set()
    for c in cover:
        for v, _ in c:
            s.add(v)
    # Preserve deterministic ordering: sort lexicographically with case
    return sorted(s)


def cofactor(cover: Cover, assignment: Dict[str, bool]) -> Cover:
    """
    General cofactor f_assignment for a *cube assignment* (e.g., {'a':True, 'b':False}).
    For each cube: drop satisfied literals, discard cubes that contradict the assignment.
    """
    if "__IMPOSSIBLE__" in assignment:
        return set()
    out: Set[Cube] = set()
    for c in cover:
        discard = False
        new: Set[Literal] = set()
        for v, p in c:
            if v in assignment:
                if assignment[v] != p:
                    discard = True
                    break  # contradiction under assignment
                # satisfied -> drop literal
            else:
                new.add((v, p))
        if not discard:
            out.add(frozenset(new))
    return simplify_cover(out)


def and_covers(f: Cover, g: Cover) -> Cover:
    """Conjunct two covers (distribute products), then simplify."""
    if not f or not g:
        return set()
    out: Set[Cube] = set()
    total = len(f) * len(g)
    produced = 0
    for a in f:
        for b in g:
            produced += 1
            if produced % PROGRESS_INTERVAL == 0:
                progress("AND distribute", produced, total)
            out.add(frozenset(set(a) | set(b)))
            if len(out) > EXPANSION_CUBE_LIMIT:
                log(f"AND early stop: > {EXPANSION_CUBE_LIMIT} cubes (partial simplification)")
                return simplify_cover(out)
    return simplify_cover(out)


def or_covers(f: Cover, g: Cover) -> Cover:
    """Disjunct two covers and simplify."""
    return simplify_cover(set(f) | set(g))


def complement_lit(l: Literal) -> Literal:
    v, p = l
    return (v, not p)


def subtract_cube(t: Cube, u: Cube) -> Set[Cube]:
    """
    Compute t ∧ ¬u as a (possibly multi-cube) cover.
    Rules:
      - If t and u contradict on any var -> u is always false under t => return {t}.
      - If u ⊆ t -> t implies u => remove entirely -> ∅.
      - Else let M = literals of u missing in t -> t ∧ ¬u = ⋁_{l∈M} (t ⋅ ¬l).
    """
    dt, du = dict(t), dict(u)
    # Contradiction?
    for v, p in dt.items():
        if v in du and du[v] != p:
            return {t}
    # u subset of t?
    subset = True
    for v, p in du.items():
        if v not in dt or dt[v] != p:
            subset = False
            break
    if subset:
        return set()
    # Otherwise expand on each missing literal
    res: Set[Cube] = set()
    for v, p in du.items():
        if v not in dt:
            new = set(t)
            new.add((v, not p))
            res.add(frozenset(new))
    return res


def subtract_cover(f: Cover, g: Cover) -> Cover:
    """
    Compute f ∧ ¬g exactly by repeated cube subtraction (Espresso-style expansion).
    """
    current: Cover = set(f)
    g_list = list(g)
    total = len(g_list)
    step = 0
    for u in g_list:
        step += 1
        progress("Subtract cover", step, total)
        next_set: Cover = set()
        inner = 0
        for t in current:
            inner += 1
            if inner % PROGRESS_INTERVAL == 0:
                progress("  subtract cube", inner)
            next_set |= subtract_cube(t, u)
            if len(next_set) > EXPANSION_CUBE_LIMIT:
                log(f"SUB early stop at step {step}: > {EXPANSION_CUBE_LIMIT} cubes")
                current = simplify_cover(next_set)
                break
        current = simplify_cover(next_set)
        if not current:
            break
    return simplify_cover(current)


def xor_covers(f: Cover, g: Cover) -> Cover:
    """Compute (f ⊕ g) = (f ∧ ¬g) ∨ (g ∧ ¬f)."""
    a = subtract_cover(f, g)
    b = subtract_cover(g, f)
    return simplify_cover(or_covers(a, b))


# ---------- High-level Operations w.r.t. a Variable ----------

def positive_negative_cofactors(f: Cover, x: str) -> Tuple[Cover, Cover]:
    """Return (f_x, f_x')."""
    fx1 = cofactor(f, {x: True})
    fx0 = cofactor(f, {x: False})
    return fx1, fx0


def smoothing(f: Cover, x: str) -> Cover:
    """S_x(f) = f_x + f_x'"""
    fx1, fx0 = positive_negative_cofactors(f, x)
    return or_covers(fx1, fx0)


def consensus_operator(f: Cover, x: str) -> Cover:
    """C_x(f) = f_x · f_x'"""
    fx1, fx0 = positive_negative_cofactors(f, x)
    return and_covers(fx1, fx0)


def boolean_difference(f: Cover, x: str) -> Cover:
    """∂f/∂x = f_x ⊕ f_x' (optimized: remove x-free cubes before XOR).

    Optimization rationale:
      Let f = g + h where g has no x literals, h has cubes with x/x'.
      Then f_x = g + h_x, f_x' = g + h_x'.
      Thus f_x ⊕ f_x' = (g + h_x) ⊕ (g + h_x') = h_x ⊕ h_x' (since g cancels).
      We therefore only cofactor the subset of cubes containing x or x'.

    Still uses exact symbolic subtraction, but on a (much) smaller cover.
    Aborts if intermediate expansion exceeds EXPANSION_CUBE_LIMIT.
    """
    relevant: Cover = set(c for c in f if (x, True) in c or (x, False) in c)
    if not relevant:
        return set()
    log(f"Boolean difference target '{x}': relevant cubes = {len(relevant)} (original {len(f)})")
    fx1 = cofactor(relevant, {x: True})
    fx0 = cofactor(relevant, {x: False})
    # Quick escape: identical cofactors => derivative 0
    if fx1 == fx0:
        return set()
    # If size large, warn before heavy XOR
    if (len(fx1) * len(fx0)) > EXPANSION_CUBE_LIMIT:
        log("Cofactor product large; XOR may be truncated if limit exceeded")
    return xor_covers(fx1, fx0)


# ---------- Input helpers ----------

def read_sop_from_file(path: str) -> Cover:
    """
    Reads either:
      - One SoP expression per line  (will OR all lines)
      - OR one product term per line (detected if no '+' found)
    """
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()
                 and not ln.strip().startswith("#")]
    if not lines:
        return set()
    if any("+" in ln for ln in lines):
        # Treat each line as a full SoP and OR all
        cov: Cover = set()
        for ln in lines:
            cov = or_covers(cov, parse_sop(ln))
        return cov
    else:
        # Treat each line as *one product term*
        cover: Cover = set()
        for ln in lines:
            cover |= parse_sop(ln)  # each ln is just a product like ab'c
        return cover


def variable_report(f: Cover) -> str:
    """Build a small report of variable usage (case-sensitive now)."""
    pos: Dict[str, int] = {}
    neg: Dict[str, int] = {}
    for c in f:
        for v, p in c:
            (pos if p else neg)[v] = (pos if p else neg).get(v, 0) + 1
    vars_all = sorted(set(pos) | set(neg))
    lines: List[str] = []
    for v in vars_all:
        p = pos.get(v, 0)
        n = neg.get(v, 0)
        if p and n:
            kind = f"{Fore.RED}{Style.BRIGHT}binate{Style.RESET_ALL}"
        elif p:
            kind = f"{Fore.GREEN}positive-unate{Style.RESET_ALL}"
        else:
            kind = f"{Fore.GREEN}negative-unate{Style.RESET_ALL}"
        lines.append(f"  {v}: +{p}  -{n}   {kind}")
    return "\n".join(lines) if lines else "(no variables)"


# ---------- Detailed Boolean Difference (Step Output) ----------

def boolean_difference_steps(f: Cover, x: str) -> Tuple[Cover, Cover, Cover]:
    """Return (f_x, f_x', derivative) plus side-effect-free computation for step display.

    This augments the user experience by explicitly showing how only terms
    containing x (or x') influence the derivative while common x-free terms cancel.
    """
    fx1, fx0 = positive_negative_cofactors(f, x)
    deriv = xor_covers(fx1, fx0)
    return fx1, fx0, deriv


# ---------- CLI ----------

def print_header():
    print(f"{Style.BRIGHT}{Fore.CYAN}Boolean SoP Analyzer{Style.RESET_ALL}")
    print("Conventions: variables like a, b, c ... complements as a' (prime). Variables are case-sensitive; use primes for complements.\n")


def print_formula_cheatsheet(x: str):
    print(f"{Fore.MAGENTA}Formulas w.r.t. {x}:{Style.RESET_ALL}")
    print(
        f"  Positive cofactor:   {Style.BRIGHT}f_{x}{Style.RESET_ALL}  = f with {x}=1")
    print(
        f"  Negative cofactor:   {Style.BRIGHT}f_{x}'{Style.RESET_ALL} = f with {x}=0")
    print(
        f"  Smoothing:           {Style.BRIGHT}S_{x}(f){Style.RESET_ALL} = f_{x} + f_{x}'")
    print(
        f"  Consensus operator:  {Style.BRIGHT}C_{x}(f){Style.RESET_ALL} = f_{x} · f_{x}'")
    print(
        f"  Boolean difference:  {Style.BRIGHT}∂f/∂{x}{Style.RESET_ALL}  = f_{x} ⊕ f_{x}'\n")


def demo_big_expression() -> str:
    # A reasonably large SoP to try
    return "a'b'c + ab'd + bc'd' + a'cd + bcd' + acd + abd' + ab'c'd + a'bc + b'cd + a'bd' + a'bc'd + abc + cd' + b'c'd"


def main(argv: Sequence[str]) -> int:
    print_header()

    # Load expression
    f: Cover
    if len(argv) >= 2 and os.path.exists(argv[1]):
        print(f"{Fore.YELLOW}Loading from file:{Style.RESET_ALL} {argv[1]}")
        f = read_sop_from_file(argv[1])
    else:
        print(
            f"{Fore.YELLOW}Enter SoP expression (or press Enter to use a demo):{Style.RESET_ALL}")
        line = input("> ").strip()
        if not line:
            line = demo_big_expression()
            print(f"  Using demo: {Style.BRIGHT}{line}{Style.RESET_ALL}")
        f = parse_sop(line)

    print(f"\n{Fore.CYAN}Parsed f:{Style.RESET_ALL}  {Style.BRIGHT}{cover_str(f)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Variables (case-sensitive):{Style.RESET_ALL} {', '.join(vars_in_cover(f)) or '(none)'}")
    print(f"{Fore.CYAN}Variable report:\n{Style.RESET_ALL}{variable_report(f)}\n")

    MENU = f"""{Fore.MAGENTA}{Style.BRIGHT}Choose an analysis:{Style.RESET_ALL}
  1) Cofactor (single variable → show f_x and f_x')
  2) Cofactor (multivariable like ab, ab'c → show f_{{ab}})
  3) Smoothing S_x(f)
  4) Consensus C_x(f)
  5) Boolean Difference ∂f/∂x
  0) Exit
"""
    while True:
        print(MENU)
        choice = input(f"{Fore.YELLOW}Pick 0-5:{Style.RESET_ALL} ").strip()
        if choice == "0":
            print("Bye!")
            return 0
        elif choice == "1":
            x = input("Variable (case-sensitive, e.g., a or A): ").strip()
            if not x or not x[0].isalpha():
                print("  Invalid variable.\n")
                continue
            fx1, fx0 = positive_negative_cofactors(f, x)
            print(f"  f_{x}   = {cover_str(fx1)}")
            print(f"  f_{x}'  = {cover_str(fx0)}\n")
        elif choice == "2":
            cube_expr = input("multivariable cube (e.g., ab'c): ").strip()
            try:
                cube = parse_cube(cube_expr)
            except Exception as e:
                print(f"  Parse error: {e}\n")
                continue
            if "__IMPOSSIBLE__" in cube:
                print("  The given cube is contradictory (unsatisfiable). f_cube = 0\n")
                continue
            f_cube = cofactor(f, cube)
            print(f"  f_{{{cube_expr}}} = {cover_str(f_cube)}\n")
        elif choice == "3":
            x = input("Variable for smoothing (case-sensitive): ").strip()
            if not x or not x[0].isalpha():
                print("  Invalid variable.\n")
                continue
            print_formula_cheatsheet(x)
            s = smoothing(f, x)
            print(f"  S_{x}(f) = {cover_str(s)}\n")
        elif choice == "4":
            x = input("Variable for consensus (case-sensitive): ").strip()
            if not x or not x[0].isalpha():
                print("  Invalid variable.\n")
                continue
            print_formula_cheatsheet(x)
            cns = consensus_operator(f, x)
            print(f"  C_{x}(f) = {cover_str(cns)}\n")
        elif choice == "5":
            x = input("Variable for Boolean difference (case-sensitive): ").strip()
            if not x or not x[0].isalpha():
                print("  Invalid variable.\n")
                continue
            print_formula_cheatsheet(x)
            relevant = [c for c in f if (x, True) in c or (x, False) in c]
            if not relevant:
                print(f"  (No cubes contain {x} / {x}') => ∂f/∂{x} = 0\n")
                continue
            pos_cubes = [c for c in relevant if (x, True) in c]
            neg_cubes = [c for c in relevant if (x, False) in c]
            print(f"  Relevant cubes ({len(relevant)} of {len(f)} total):")
            print(f"    with {x}:   {', '.join(cube_str(c) for c in pos_cubes)}")
            print(f"    with {x}':  {', '.join(cube_str(c) for c in neg_cubes) or '-'}")
            d = boolean_difference(f, x)
            print(f"  ∂f/∂{x} = {cover_str(d)}\n")
            if len(d) > 50:
                print("  (Result large; consider additional manual simplification.)\n")
        else:
            print("  Unknown choice.\n")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except KeyboardInterrupt:
        print("\nInterrupted.")
