#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boolean SoP Analyzer (Optimized Version)
========================================
Adds an optimized Boolean difference computation that avoids
the combinational blow-up seen in the generic XOR path by:
  - Restricting to cubes containing the differentiation variable.
  - Optionally emitting a compact symbolic XOR form instead of
    fully expanding to a (possibly huge) OR-of-products.

Configuration:
  UPPERCASE_MEANS_COMPLEMENT = True  (original behavior)
    If False, 'A' is treated the same as 'a' (positive literal) unless primed.
  DIFFERENCE_KEEP_SYMBOLIC_XOR = True
    If True, derivative w.r.t x is returned (and displayed) as
      (PosPart) ⊕ (NegPart)
    rather than fully expanded SoP.

Original functionality preserved unless you choose the optimized derivative.

"""

from __future__ import annotations
import sys, os, re
from typing import Dict, List, Sequence, Set, Tuple, FrozenSet

try:
    from colorama import init as colorama_init, Fore, Style
except Exception:
    class Dummy: RESET_ALL = ""
    class ForeDummy(Dummy):
        CYAN=GREEN=MAGENTA=YELLOW=RED=BLUE=""
    class StyleDummy(Dummy):
        BRIGHT=DIM=NORMAL=""
    Fore=ForeDummy()
    Style=StyleDummy()
    def colorama_init(): pass

colorama_init()

# ---------------- Configuration ----------------
UPPERCASE_MEANS_COMPLEMENT = True
DIFFERENCE_KEEP_SYMBOLIC_XOR = True
# Safety threshold: if intermediate expansion grows beyond this many cubes
# we abort further expansion (leave partially simplified).
EXPANSION_CUBE_LIMIT = 5000

# (var, True) = positive literal, (var, False) = complemented
Literal = Tuple[str, bool]
Cube    = FrozenSet[Literal]
Cover   = Set[Cube]

# -------------- Parsing & Formatting -----------

def normalize_var(token: str) -> Tuple[str, bool]:
    t = token.strip()
    if not t: raise ValueError("Empty token")
    prime = t.endswith("'")
    base  = t.rstrip("'")
    if not base: raise ValueError(f"Bad literal: {token}")
    v = base[0].lower() + base[1:]
    if UPPERCASE_MEANS_COMPLEMENT and base[0].isupper() and not prime:
        return v, False
    return v, (not prime)

def parse_sop(expr: str) -> Cover:
    expr = expr.strip()
    if expr == "0": return set()
    if expr == "1": return {frozenset()}
    terms = [t.strip() for t in expr.replace(" ", "").split("+")]
    cover: Cover = set()
    for term in terms:
        if term in ("","0"): continue
        if term == "1":
            cover.add(frozenset()); continue
        lits: List[Literal] = []
        i=0
        while i < len(term):
            ch = term[i]
            if not ch.isalpha(): raise ValueError(f"Bad char '{ch}' in term '{term}'")
            j=i+1
            while j < len(term) and term[j].isdigit(): j+=1
            prime=False
            if j < len(term) and term[j]=="'":
                prime=True; j+=1
            token = term[i:j]
            v,pol = normalize_var(token + ("'" if prime else ""))
            lits.append((v,pol))
            i=j
        cover.add(frozenset(lits))
    return simplify_cover(cover)

def parse_cube(expr: str) -> Dict[str,bool]:
    expr = expr.strip().replace(" ","")
    if expr in ("","1"): return {}
    if expr == "0": return {"__IMPOSSIBLE__": True}
    assign: Dict[str,bool]={}
    i=0
    while i < len(expr):
        ch=expr[i]
        if not ch.isalpha(): raise ValueError(f"Bad char '{ch}' in cube '{expr}'")
        j=i+1
        while j < len(expr) and expr[j].isdigit(): j+=1
        prime=False
        if j < len(expr) and expr[j]=="'":
            prime=True; j+=1
        token=expr[i:j]
        v,pol=normalize_var(token + ("'" if prime else ""))
        if v in assign and assign[v]!=pol:
            return {"__IMPOSSIBLE__": True}
        assign[v]=pol
        i=j
    return assign

def lit_str(var: str, pol: bool) -> str:
    return f"{var}{'' if pol else "'"}"

def cube_str(c: Cube) -> str:
    if not c: return "1"
    return "".join(lit_str(v,p) for v,p in sorted(c))

def cover_str(cover: Cover) -> str:
    if not cover: return "0"
    return " + ".join(cube_str(c) for c in sorted(cover, key=lambda x:(len(x),sorted(x))))

# -------------- Core Cover Algebra -------------

def simplify_cover(cover: Cover) -> Cover:
    # remove contradictory literals & duplicates
    cleaned:Set[Cube]=set()
    for c in cover:
        d:Dict[str,bool]={}
        ok=True
        for v,p in c:
            if v in d and d[v]!=p:
                ok=False; break
            d[v]=p
        if ok:
            cleaned.add(frozenset(d.items()))
    # absorption
    minimal=set(cleaned)
    for c in cleaned:
        for u in cleaned:
            if u is c: continue
            if u.issubset(c):
                minimal.discard(c)
                break
    return minimal

def vars_in_cover(cover: Cover) -> List[str]:
    s:set[str]=set()
    for c in cover:
        for v,_ in c: s.add(v)
    return sorted(s)

def cofactor(cover: Cover, assignment: Dict[str,bool]) -> Cover:
    if "__IMPOSSIBLE__" in assignment: return set()
    out:Set[Cube]=set()
    for c in cover:
        new:Set[Literal]=set()
        discard=False
        for v,p in c:
            if v in assignment:
                if assignment[v]!=p:
                    discard=True; break
                # satisfied literal dropped
            else:
                new.add((v,p))
        if not discard:
            out.add(frozenset(new))
    return simplify_cover(out)

def and_covers(f: Cover, g: Cover) -> Cover:
    if not f or not g: return set()
    out:Set[Cube]=set()
    for a in f:
        for b in g:
            out.add(frozenset(set(a)|set(b)))
            if len(out) > EXPANSION_CUBE_LIMIT:
                return simplify_cover(out)
    return simplify_cover(out)

def or_covers(f: Cover, g: Cover) -> Cover:
    return simplify_cover(set(f)|set(g))

def subtract_cube(t: Cube, u: Cube) -> Set[Cube]:
    dt,du = dict(t), dict(u)
    # contradiction -> u false under t
    for v,p in dt.items():
        if v in du and du[v]!=p:
            return {t}
    # subset?
    for v,p in du.items():
        if v not in dt or dt[v]!=p:
            break
    else:
        return set()
    # expand on missing literals (limit)
    res:Set[Cube]=set()
    for v,p in du.items():
        if v not in dt:
            new=set(t)
            new.add((v, not p))
            res.add(frozenset(new))
            if len(res) > EXPANSION_CUBE_LIMIT:
                return res
    return res

def subtract_cover(f: Cover, g: Cover) -> Cover:
    current=set(f)
    for u in g:
        nxt:Cover=set()
        for t in current:
            nxt |= subtract_cube(t,u)
            if len(nxt) > EXPANSION_CUBE_LIMIT:
                current = simplify_cover(nxt)
                break
        current = simplify_cover(nxt)
        if not current:
            break
    return simplify_cover(current)

def xor_covers(f: Cover, g: Cover) -> Cover:
    # (f ∧ ¬g) ∨ (g ∧ ¬f)
    return simplify_cover(or_covers(subtract_cover(f,g), subtract_cover(g,f)))

# -------------- High-level (original) -----------

def positive_negative_cofactors(f: Cover, x: str) -> Tuple[Cover, Cover]:
    return cofactor(f,{x:True}), cofactor(f,{x:False})

def smoothing(f: Cover, x: str) -> Cover:
    fx1,fx0 = positive_negative_cofactors(f,x)
    return or_covers(fx1,fx0)

def consensus_operator(f: Cover, x: str) -> Cover:
    fx1,fx0 = positive_negative_cofactors(f,x)
    return and_covers(fx1,fx0)

def boolean_difference_naive(f: Cover, x: str) -> Cover:
    fx1,fx0 = positive_negative_cofactors(f,x)
    return xor_covers(fx1,fx0)

# -------------- Optimized Boolean Difference ----

def extract_relevant_for_var(f: Cover, x: str) -> Cover:
    """Keep only cubes that mention x (pos or neg)."""
    out:Set[Cube]=set()
    for c in f:
        for v,_ in c:
            if v == x:
                out.add(c)
                break
    return out

def cofactors_restricted(f_reduced: Cover, x: str) -> Tuple[Cover, Cover]:
    """Given only cubes containing x/x', strip the literal and separate pos/neg parts."""
    pos:Set[Cube]=set()
    neg:Set[Cube]=set()
    for c in f_reduced:
        new_lits=[]
        pos_here=False
        neg_here=False
        for v,p in c:
            if v==x:
                if p: pos_here=True
                else: neg_here=True
            else:
                new_lits.append((v,p))
        cube_new=frozenset(new_lits)
        if pos_here and neg_here:
            # (Should not happen in cleaned cubes) -> contradictory, skip
            continue
        if pos_here:
            pos.add(cube_new)
        else:
            neg.add(cube_new)
    return simplify_cover(pos), simplify_cover(neg)

def boolean_difference_var_optimized(f: Cover, x: str, keep_symbolic: bool = DIFFERENCE_KEEP_SYMBOLIC_XOR) -> Tuple[str, Cover]:
    """
    Returns (display_string, expanded_cover_if_available).
    If keep_symbolic is True, display_string is a symbolic XOR form without full expansion.
    """
    relevant = extract_relevant_for_var(f,x)
    if not relevant:
        return f"0  (x not present)", set()
    pos, neg = cofactors_restricted(relevant, x)
    if keep_symbolic:
        pos_s = cover_str(pos) or "0"
        neg_s = cover_str(neg) or "0"
        if pos_s=="0" and neg_s=="0":
            return "0", set()
        if pos_s=="0":  # 0 ⊕ N = N
            return neg_s, neg
        if neg_s=="0":  # P ⊕ 0 = P
            return pos_s, pos
        symbolic = f"({pos_s}) ⊕ ({neg_s})"
        return symbolic, set()  # we intentionally skip huge expansion
    # Else expand explicitly (might still blow up on pathological cases)
    expanded = xor_covers(pos, neg)
    return cover_str(expanded), expanded

# -------------- Reporting -----------------------

def variable_report(f: Cover) -> str:
    pos_ct:Dict[str,int]={}
    neg_ct:Dict[str,int]={}
    for c in f:
        for v,p in c:
            if p: pos_ct[v]=pos_ct.get(v,0)+1
            else: neg_ct[v]=neg_ct.get(v,0)+1
    all_vars=sorted(set(pos_ct)|set(neg_ct))
    lines=[]
    for v in all_vars:
        p = pos_ct.get(v,0)
        n = neg_ct.get(v,0)
        if p and n:
            kind=f"{Fore.RED}{Style.BRIGHT}binate{Style.RESET_ALL}"
        elif p:
            kind=f"{Fore.GREEN}positive-unate{Style.RESET_ALL}"
        else:
            kind=f"{Fore.GREEN}negative-unate{Style.RESET_ALL}"
        lines.append(f"  {v}: +{p}  -{n}   {kind}")
    return "\n".join(lines) if lines else "(no variables)"

# -------------- I/O Helpers ---------------------

def read_sop_from_file(path: str) -> Cover:
    with open(path,"r",encoding="utf-8") as fh:
        lines=[ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
    if not lines: return set()
    if any("+" in ln for ln in lines):
        cov:set[Cube]=set()
        for ln in lines:
            cov = or_covers(cov, parse_sop(ln))
        return cov
    cover:set[Cube]=set()
    for ln in lines:
        cover |= parse_sop(ln)
    return cover

# -------------- CLI Utilities -------------------

def print_header():
    print(f"{Style.BRIGHT}{Fore.CYAN}Boolean SoP Analyzer (Optimized){Style.RESET_ALL}")
    print("Conventions: variables a,b,c; complements as a'.")
    if UPPERCASE_MEANS_COMPLEMENT:
        print("NOTE: Uppercase without prime is treated as complemented (A == a').\n")
    else:
        print("NOTE: Upper/lowercase are the same (A == a). Use prime for complement.\n")

def print_formula_cheatsheet(x: str):
    print(f"{Fore.MAGENTA}Formulas w.r.t. {x}:{Style.RESET_ALL}")
    print(f"  f_{x}   : positive cofactor  (x=1)")
    print(f"  f_{x}'  : negative cofactor  (x=0)")
    print(f"  S_{x}(f)= f_{x} + f_{x}'")
    print(f"  C_{x}(f)= f_{x} · f_{x}'")
    print(f"  ∂f/∂{x} = f_{x} ⊕ f_{x}'\n")

def demo_big_expression()->str:
    return "a'b'c + ab'd + bc'd' + a'cd + bcd' + acd + abd' + ab'c'd + a'bc + b'cd + a'bd' + a'bc'd + abc + cd' + b'c'd"

# -------------- Main CLI ------------------------

MENU = f"""{Fore.MAGENTA}{Style.BRIGHT}Choose an analysis:{Style.RESET_ALL}
  1) Cofactor (single variable → f_x and f_x')
  2) Cofactor (multi-variable cube like ab'c → f_{cube})
  3) Smoothing S_x(f)
  4) Consensus C_x(f)
  5) Boolean Difference ∂f/∂x (optimized, symbolic if enabled)
  6) Boolean Difference ∂f/∂x (original naive expansion)
  7) Show formulas for a variable
  0) Exit
"""

def main(argv: Sequence[str]) -> int:
    print_header()
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

    print(f"\n{Fore.CYAN}Parsed f:{Style.RESET_ALL} {Style.BRIGHT}{cover_str(f)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Variables:{Style.RESET_ALL} {', '.join(vars_in_cover(f)) or '(none)'}")
    print(f"{Fore.CYAN}Variable report:\n{Style.RESET_ALL}{variable_report(f)}\n")

    while True:
        print(MENU)
        choice = input(f"{Fore.YELLOW}Pick 0-7:{Style.RESET_ALL} ").strip()
        if choice == "0":
            print("Bye!")
            return 0
        elif choice == "1":
            x = input("Variable: ").strip().lower()
            if not x or not x[0].isalpha(): print("  Invalid.\n"); continue
            fx1,fx0 = positive_negative_cofactors(f,x)
            print(f"  f_{x}   = {cover_str(fx1)}")
            print(f"  f_{x}'  = {cover_str(fx0)}\n")
        elif choice == "2":
            cube_expr = input("Cube (e.g. ab'c): ").strip()
            try:
                cube = parse_cube(cube_expr)
            except Exception as e:
                print(f"  Parse error: {e}\n"); continue
            if "__IMPOSSIBLE__" in cube:
                print("  Contradictory cube => f_cube = 0\n"); continue
            fc = cofactor(f,cube)
            print(f"  f_{{{cube_expr}}} = {cover_str(fc)}\n")
        elif choice == "3":
            x = input("Variable for smoothing: ").strip().lower()
            if not x or not x[0].isalpha(): print("  Invalid.\n"); continue
            print_formula_cheatsheet(x)
            s = smoothing(f,x)
            print(f"  S_{x}(f) = {cover_str(s)}\n")
        elif choice == "4":
            x = input("Variable for consensus: ").strip().lower()
            if not x or not x[0].isalpha(): print("  Invalid.\n"); continue
            print_formula_cheatsheet(x)
            cns = consensus_operator(f,x)
            print(f"  C_{x}(f) = {cover_str(cns)}\n")
        elif choice == "5":
            x = input("Variable for Boolean difference (optimized): ").strip().lower()
            if not x or not x[0].isalpha(): print("  Invalid.\n"); continue
            print_formula_cheatsheet(x)
            display, _ = boolean_difference_var_optimized(f,x, keep_symbolic=DIFFERENCE_KEEP_SYMBOLIC_XOR)
            print(f"  ∂f/∂{x} = {display}")
            if DIFFERENCE_KEEP_SYMBOLIC_XOR:
                print("  (Symbolic XOR form kept. Set DIFFERENCE_KEEP_SYMBOLIC_XOR=False to expand.)\n")
            else:
                print()
        elif choice == "6":
            x = input("Variable for Boolean difference (naive / full): ").strip().lower()
            if not x or not x[0].isalpha(): print("  Invalid.\n"); continue
            print_formula_cheatsheet(x)
            bd = boolean_difference_naive(f,x)
            print(f"  ∂f/∂{x} (expanded) = {cover_str(bd)}\n")
        elif choice == "7":
            x = input("Variable: ").strip().lower()
            if not x or not x[0].isalpha(): print("  Invalid.\n"); continue
            print_formula_cheatsheet(x)
        else:
            print("  Unknown choice.\n")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except KeyboardInterrupt:
        print("\nInterrupted.")