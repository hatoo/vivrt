"""Side-by-side diff of every C/Rust struct pair, including double-pointer
declarations like `PrincipledGpu **materials`.

Catches the kind of silent layout bug that, once introduced, made the
addon's Mist pass come out all-zero: `LaunchParams::depth_aov` was placed
*before* `world_volume` in the C header but *after* it on the Rust side,
so the device read whichever pointer the host wrote into the wrong slot.
The visible symptom was bricks looking dramatically more saturated than
Cycles. Field-order checks like this would have caught it before render.

Exits with status 1 on any mismatch, so it can be wired into CI / a Make
target.
"""
import re
import sys
from pathlib import Path

C = Path("vibrt/src/devicecode.h").read_text(encoding="utf-8")
R = Path("vibrt/src/gpu_types.rs").read_text(encoding="utf-8")

C_STRUCTS = re.compile(r"struct\s+(\w+)\s*\{([^}]*)\}\s*;", re.DOTALL)
R_STRUCTS = re.compile(r"#\[repr\(C\)\][\s\S]*?pub struct\s+(\w+)\s*\{([^}]*)\}", re.DOTALL)

c_map = {m.group(1): m.group(2) for m in C_STRUCTS.finditer(C)}
r_map = {m.group(1): m.group(2) for m in R_STRUCTS.finditer(R)}

NAME_MAP = {"VolumeGpu": "Volume"}

# Match `<type tokens with stars/spaces> name [N];` — accept any number of stars
C_FIELD = re.compile(r"^\s*(.*?)\s+(\**)\s*(\w+)\s*(\[(\d+)\])?\s*$")

def parse_c_fields(body):
    out = []
    for line in body.split("\n"):
        line = re.sub(r"//.*", "", line).strip().rstrip(";")
        if not line:
            continue
        # Stars can hug the type or the name; collapse to "<type>* name".
        m = C_FIELD.match(line)
        if not m:
            continue
        ty = m.group(1).strip()
        stars = m.group(2)
        name = m.group(3)
        n = int(m.group(5)) if m.group(5) else None
        ty_full = (ty + stars) if stars else ty
        out.append((name, ty_full, n))
    return out

def parse_r_fields(body):
    out = []
    for line in body.split("\n"):
        line = re.sub(r"///.*|//.*", "", line).strip().rstrip(",")
        if not line.startswith("pub "):
            continue
        line = line[4:]
        m = re.match(r"(\w+):\s*(.+)", line)
        if not m:
            continue
        out.append((m.group(1), m.group(2).strip()))
    return out

mismatch_count = 0
for r_name, r_body in r_map.items():
    c_name = NAME_MAP.get(r_name, r_name)
    if c_name not in c_map:
        continue
    cf = parse_c_fields(c_map[c_name])
    rf = parse_r_fields(r_body)
    n = max(len(cf), len(rf))
    has_diff = False
    for i in range(n):
        c = cf[i] if i < len(cf) else None
        r = rf[i] if i < len(rf) else None
        c_name_field = c[0] if c else None
        r_name_field = r[0] if r else None
        if c_name_field != r_name_field:
            has_diff = True
    if has_diff or len(cf) != len(rf):
        mismatch_count += 1
        print(f"\n=== {r_name} (Rust) <-> {c_name} (C) — MISMATCH ===")
        for i in range(n):
            c = cf[i] if i < len(cf) else None
            r = rf[i] if i < len(rf) else None
            c_str = f"{c[1]} {c[0]}" + (f"[{c[2]}]" if c and c[2] else "") if c else "--"
            r_str = f"{r[0]}: {r[1]}" if r else "--"
            ok = "OK" if (c and r and c[0] == r[0]) else "XX"
            print(f"  {ok} [{i+1:>2}] {c_str:<40}  |  {r_str}")
    else:
        print(f"OK {r_name}: {len(cf)} fields match")

print(f"\n{mismatch_count} mismatch(es)")
sys.exit(1 if mismatch_count else 0)
