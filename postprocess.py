# postprocess.py
# Cleans up generated LaTeX so it actually compiles.
# This is for the bonus section.
#
# Common issues the decoder produces:
#   - unbalanced braces/brackets
#   - double superscripts like  x ^ {a} ^ {b}
#   - repeated tokens (decoder stuttering)
#   - empty arguments like  \frac {}

import re

# pairs of open/close delimiters
PAIRS = {"{": "}", "(": ")", "[": "]"}
CLOSE_MAP = {v: k for k, v in PAIRS.items()}

# LaTeX commands that take a braced argument
CMDS_WITH_ARG = [
    r"\\frac", r"\\sqrt", r"\\hat", r"\\bar", r"\\tilde",
    r"\\vec", r"\\dot", r"\\ddot", r"\\overline", r"\\underline",
    r"\\mathbf", r"\\mathit", r"\\mathrm", r"\\mathcal", r"\\mathbb",
    r"\\text", r"\\operatorname",
]


def fix_braces(text):
    """Balance { } ( ) [ ] using a stack."""
    stack = []
    out = []
    for ch in text:
        if ch in PAIRS:
            stack.append(ch)
            out.append(ch)
        elif ch in CLOSE_MAP:
            if stack and stack[-1] == CLOSE_MAP[ch]:
                stack.pop()
                out.append(ch)
            # else: unmatched closer, just skip it
        else:
            out.append(ch)
    # append missing closers
    while stack:
        out.append(PAIRS[stack.pop()])
    return "".join(out)


def fix_empty_args(text):
    """Replace \\cmd { } with \\cmd {~} so LaTeX doesn't choke."""
    for cmd in CMDS_WITH_ARG:
        pat = cmd + r"\s*\{\s*\}"
        rep = cmd.replace("\\\\", "\\") + " {~}"
        text = re.sub(pat, rep, text)
    return text


def fix_double_scripts(text):
    """Merge  ^ {a} ^ {b}  into  ^ {a ^ {b}}  (same for _)."""
    for ch in ("^", "_"):
        esc = "\\" + ch
        pat = esc + r"\s*\{([^}]*)\}\s*" + esc + r"\s*\{([^}]*)\}"
        rep = ch + r" { \1 " + ch + r" { \2 } }"
        prev = None
        while text != prev:
            prev = text
            text = re.sub(pat, rep, text)
    return text


def remove_stutters(text):
    """Collapse runs of 3+ identical tokens to just one."""
    tokens = text.split()
    result = []
    run = 0
    for i, t in enumerate(tokens):
        if i > 0 and t == tokens[i-1]:
            run += 1
        else:
            run = 0
        if run < 2:
            result.append(t)
    return " ".join(result)


def wrap_math(text):
    """Wrap in $ $ if no math delimiters are present."""
    s = text.strip()
    if not s:
        return "$  $"
    if s[0] in ("$",) or s.startswith("\\(") or s.startswith("\\[") or s.startswith("\\begin{"):
        return s
    return "$ " + s + " $"


def clean_latex(formula, add_dollars=False):
    """
    Run all the cleanup steps on one formula string.
    Set add_dollars=True if you want $ ... $ wrapping.
    """
    out = remove_stutters(formula)
    out = fix_braces(out)
    out = fix_empty_args(out)
    out = fix_double_scripts(out)
    if add_dollars:
        out = wrap_math(out)
    out = re.sub(r" {2,}", " ", out).strip()
    return out


# test it
if __name__ == "__main__":
    tests = [
        r"\frac { x } { }",
        r"\frac { { a + b }",
        r"x ^ { 2 } ^ { 3 }",
        r"a _ { i } _ { j }",
        r"\sum \sum \sum \sum _ { i }",
        r"( [ a + b }",
    ]
    for t in tests:
        print("IN :", t)
        print("OUT:", clean_latex(t, add_dollars=True))
        print()
