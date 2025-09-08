#!/usr/bin/env python3
"""
Boolean SoP Analyzer - Exact analyzer for Boolean expressions in Sum of Products form.

Features:
- Cofactors w.r.t. variables or cubes (e.g., a, ab, ab')
- Boolean Difference, Consensus, and Smoothing operations
- Case-sensitive variables; complements use prime notation (a')
- Input: keyboard or file; Output: same prime style
"""

from __future__ import annotations
import sys, os, re
from typing import Dict, List, Optional, Sequence, Set, Tuple, FrozenSet

try:
    from colorama import init as colorama_init, Fore, Style
except ImportError:
    class _Dummy:
        RESET_ALL = CYAN = GREEN = MAGENTA = YELLOW = RED = BLUE = ""
        BRIGHT = DIM = NORMAL = ""
    Fore = Style = _Dummy()
    def colorama_init(): pass

colorama_init()

Literal = Tuple[str, bool]  # (var, True) means var, (var, False) means var'
Cube = FrozenSet[Literal]          # product term
Cover = Set[Cube]                  # SoP: OR of cubes

VERBOSE = True
PROGRESS_INTERVAL = 250
EXPANSION_CUBE_LIMIT = 4000

def log(msg: str):
    if VERBOSE: print(f"[info] {msg}")

def progress(kind: str, count: int, total: Optional[int] = None):
    if VERBOSE and count % PROGRESS_INTERVAL == 0:
        suffix = f"{count}/{total}" if total else str(count)
        print(f"[working] {kind}: {suffix} ...", flush=True)


def normalize_var(token: str) -> Tuple[str, bool]:
    """Convert literal token to (name, polarity). Case-sensitive; prime indicates complement."""
    t = token.strip()
    if not t:
        raise ValueError("Empty token")
    prime = t.endswith("'")
    base = t.rstrip("'")
    if not base:
        raise ValueError(f"Bad literal: {token}")
    return base, (not prime)


def parse_sop(expr: str) -> Cover:
    """Parse SoP string: e.g. "ab + a'c + bcd". '+' separates terms."""
    expr = expr.strip()
    if expr == "0": return set()
    if expr == "1": return {frozenset()}
    terms = [t.strip() for t in expr.replace(" ", "").split("+")]
    cover: Cover = set()
    for term in terms:
        if term == "" or term == "0": continue
        if term == "1": 
            cover.add(frozenset())
            continue
        lits: List[Literal] = []
        i = 0
        while i < len(term):
            ch = term[i]
            if ch == "+":
                raise ValueError(f"Unexpected '+' in term '{term}' at position {i}")
            if not ch.isalpha():
                raise ValueError(f"Bad character '{ch}' in term '{term}'")
            j = i + 1
            while j < len(term) and term[j].isdigit(): j += 1
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
    """Parse cube like "ab'c" into assignment dict {'a':True, 'b':False, 'c':True}."""
    expr = expr.strip().replace(" ", "")
    if expr in ("", "1"): return {}
    if expr == "0": return {"__IMPOSSIBLE__": True}
    assign: Dict[str, bool] = {}
    i = 0
    while i < len(expr):
        ch = expr[i]
        if not ch.isalpha():
            raise ValueError(f"Bad character '{ch}' in cube '{expr}'")
        j = i + 1
        while j < len(expr) and expr[j].isdigit(): j += 1
        prime = False
        if j < len(expr) and expr[j] == "'":
            prime = True
            j += 1
        token = expr[i:j]
        v, pol = normalize_var(token + ("'" if prime else ""))
        if v in assign and assign[v] != pol:
            return {"__IMPOSSIBLE__": True}
        assign[v] = pol
        i = j
    return assign


def lit_str(var: str, pol: bool) -> str:
    """Format literal as var or var'."""
    return f"{var}{'' if pol else "'"}"

def cube_str(c: Cube) -> str:
    """Format cube as concatenated literals, or '1' for empty cube."""
    return "1" if len(c) == 0 else "".join(lit_str(v, p) for v, p in sorted(c))

def cover_str(cover: Cover) -> str:
    """Format cover as 'term + term + ...' or '0' for empty cover."""
    if not cover: return "0"
    return " + ".join(cube_str(c) for c in sorted(cover, key=lambda x: (len(x), sorted(x))))


# ---------- Core Boolean Cover Algebra ----------

def simplify_cover(cover: Cover) -> Cover:
    """Simplify via contradiction removal, deduplication, and absorption."""
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
    
    minimal: Set[Cube] = set(cleaned)
    for c in cleaned:
        for u in cleaned:
            if u is not c and u.issubset(c):
                minimal.discard(c)
                break
    return minimal


def vars_in_cover(cover: Cover) -> List[str]:
    """Get sorted list of all variables in cover."""
    s: Set[str] = set()
    for c in cover:
        for v, _ in c:
            s.add(v)
    return sorted(s)


def cofactor(cover: Cover, assignment: Dict[str, bool]) -> Cover:
    """General cofactor for cube assignment. Drop satisfied literals, discard contradictory cubes."""
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
                    break
            else:
                new.add((v, p))
        if not discard:
            out.add(frozenset(new))
    return simplify_cover(out)


def and_covers(f: Cover, g: Cover) -> Cover:
    """Conjunct two covers (distribute products), then simplify."""
    if not f or not g: return set()
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
                log(f"AND early stop: > {EXPANSION_CUBE_LIMIT} cubes")
                return simplify_cover(out)
    return simplify_cover(out)

def or_covers(f: Cover, g: Cover) -> Cover:
    """Disjunct two covers and simplify."""
    return simplify_cover(set(f) | set(g))

def complement_lit(l: Literal) -> Literal:
    """Return complemented literal."""
    v, p = l
    return (v, not p)


def subtract_cube(t: Cube, u: Cube) -> Set[Cube]:
    """Compute t ∧ ¬u as a cover."""
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
    if subset: return set()
    # Expand on each missing literal
    res: Set[Cube] = set()
    for v, p in du.items():
        if v not in dt:
            new = set(t)
            new.add((v, not p))
            res.add(frozenset(new))
    return res


def subtract_cover(f: Cover, g: Cover) -> Cover:
    """Compute f ∧ ¬g exactly by repeated cube subtraction."""
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
        if not current: break
    return simplify_cover(current)

def xor_covers(f: Cover, g: Cover) -> Cover:
    """Compute (f ⊕ g) = (f ∧ ¬g) ∨ (g ∧ ¬f)."""
    a = subtract_cover(f, g)
    b = subtract_cover(g, f)
    return simplify_cover(or_covers(a, b))


# ---------- High-level Operations w.r.t. a Variable ----------

def positive_negative_cofactors(f: Cover, x: str) -> Tuple[Cover, Cover]:
    """Return (f_x, f_x')."""
    return cofactor(f, {x: True}), cofactor(f, {x: False})

def smoothing(f: Cover, x: str) -> Cover:
    """S_x(f) = f_x + f_x'"""
    fx1, fx0 = positive_negative_cofactors(f, x)
    return or_covers(fx1, fx0)

def consensus_operator(f: Cover, x: str) -> Cover:
    """C_x(f) = f_x · f_x'"""
    fx1, fx0 = positive_negative_cofactors(f, x)
    return and_covers(fx1, fx0)


def boolean_difference(f: Cover, x: str) -> Cover:
    """∂f/∂x = f_x ⊕ f_x' (optimized: only process cubes containing x)."""
    relevant: Cover = set(c for c in f if (x, True) in c or (x, False) in c)
    if not relevant: return set()
    log(f"Boolean difference '{x}': relevant cubes = {len(relevant)} (original {len(f)})")
    fx1 = cofactor(relevant, {x: True})
    fx0 = cofactor(relevant, {x: False})
    if fx1 == fx0: return set()
    if (len(fx1) * len(fx0)) > EXPANSION_CUBE_LIMIT:
        log("Cofactor product large; XOR may be truncated if limit exceeded")
    return xor_covers(fx1, fx0)




def read_sop_from_file(path: str) -> Cover:
    """Read SoP from file: one SoP per line OR one product term per line (auto-detected)."""
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
    if not lines: return set()
    
    if any("+" in ln for ln in lines):
        cov: Cover = set()
        for ln in lines:
            cov = or_covers(cov, parse_sop(ln))
        return cov
    else:
        cover: Cover = set()
        for ln in lines:
            cover |= parse_sop(ln)
        return cover


def variable_report(f: Cover) -> str:
    """Build variable usage report (case-sensitive)."""
    pos: Dict[str, int] = {}
    neg: Dict[str, int] = {}
    for c in f:
        for v, p in c:
            (pos if p else neg)[v] = (pos if p else neg).get(v, 0) + 1
    vars_all = sorted(set(pos) | set(neg))
    lines: List[str] = []
    for v in vars_all:
        p, n = pos.get(v, 0), neg.get(v, 0)
        if p and n:
            kind = f"{Fore.RED}{Style.BRIGHT}binate{Style.RESET_ALL}"
        elif p:
            kind = f"{Fore.GREEN}positive-unate{Style.RESET_ALL}"
        else:
            kind = f"{Fore.GREEN}negative-unate{Style.RESET_ALL}"
        lines.append(f"  {v}: +{p}  -{n}   {kind}")
    return "\n".join(lines) if lines else "(no variables)"





def print_header():
    print(f"{Style.BRIGHT}{Fore.CYAN}Boolean SoP Analyzer{Style.RESET_ALL}")
    print("Conventions: variables like a, b, c ... complements as a' (prime). Variables are case-sensitive; use primes for complements.\n")

def print_formula_cheatsheet(x: str):
    print(f"{Fore.MAGENTA}Formulas w.r.t. {x}:{Style.RESET_ALL}")
    print(f"  Positive cofactor:   {Style.BRIGHT}f_{x}{Style.RESET_ALL}  = f with {x}=1")
    print(f"  Negative cofactor:   {Style.BRIGHT}f_{x}'{Style.RESET_ALL} = f with {x}=0")
    print(f"  Smoothing:           {Style.BRIGHT}S_{x}(f){Style.RESET_ALL} = f_{x} + f_{x}'")
    print(f"  Consensus operator:  {Style.BRIGHT}C_{x}(f){Style.RESET_ALL} = f_{x} · f_{x}'")
    print(f"  Boolean difference:  {Style.BRIGHT}∂f/∂{x}{Style.RESET_ALL}  = f_{x} ⊕ f_{x}'\n")

def demo_big_expression() -> str:
    return "a'b'c + ab'd + bc'd' + a'cd + bcd' + acd + abd' + ab'c'd + a'bc + b'cd + a'bd' + a'bc'd + abc + cd' + b'c'd"

def get_valid_variable(prompt: str) -> Optional[str]:
    """Get and validate variable input."""
    x = input(prompt).strip()
    if not x or not x[0].isalpha():
        print("  Invalid variable.\n")
        return None
    return x


def main(argv: Sequence[str]) -> int:
    print_header()
    
    # Load expression
    if len(argv) >= 2 and os.path.exists(argv[1]):
        print(f"{Fore.YELLOW}Loading from file:{Style.RESET_ALL} {argv[1]}")
        f = read_sop_from_file(argv[1])
    else:
        print(f"{Fore.YELLOW}Enter SoP expression (or press Enter to use a demo):{Style.RESET_ALL}")
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
            x = get_valid_variable("Variable (case-sensitive, e.g., a or A): ")
            if x:
                fx1, fx0 = positive_negative_cofactors(f, x)
                print(f"  f_{x}   = {cover_str(fx1)}")
                print(f"  f_{x}'  = {cover_str(fx0)}\n")
        elif choice == "2":
            cube_expr = input("multivariable cube (e.g., ab'c): ").strip()
            try:
                cube = parse_cube(cube_expr)
                if "__IMPOSSIBLE__" in cube:
                    print("  The given cube is contradictory (unsatisfiable). f_cube = 0\n")
                else:
                    f_cube = cofactor(f, cube)
                    print(f"  f_{{{cube_expr}}} = {cover_str(f_cube)}\n")
            except Exception as e:
                print(f"  Parse error: {e}\n")
        elif choice == "3":
            x = get_valid_variable("Variable for smoothing (case-sensitive): ")
            if x:
                print_formula_cheatsheet(x)
                s = smoothing(f, x)
                print(f"  S_{x}(f) = {cover_str(s)}\n")
        elif choice == "4":
            x = get_valid_variable("Variable for consensus (case-sensitive): ")
            if x:
                print_formula_cheatsheet(x)
                cns = consensus_operator(f, x)
                print(f"  C_{x}(f) = {cover_str(cns)}\n")
        elif choice == "5":
            x = get_valid_variable("Variable for Boolean difference (case-sensitive): ")
            if x:
                print_formula_cheatsheet(x)
                relevant = [c for c in f if (x, True) in c or (x, False) in c]
                if not relevant:
                    print(f"  (No cubes contain {x} / {x}') => ∂f/∂{x} = 0\n")
                else:
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
