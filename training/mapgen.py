"""Procedural DOOM level generator (UDMF, zdoom namespace) for curriculum training.

Levels are built on a cell grid (64 map units per cell): rectangular rooms joined by corridors,
optional doors, locked doors with keycards, monsters, pickups and an exit pad. Every room is its own
sector with random floor/ceiling height, light and wall texture, so the agent never sees the same
layout twice (domain randomisation over geometry, lighting, textures and spawn sets).

Difficulty knobs live in `LevelSpec`; `spec_for_stage()` maps a curriculum stage to a spec mix that
covers the four evaluation tiers (combat, multi-room exploration, key->lock->exit, gauntlet).

Grid legend: '#' wall, '.' floor, 'D' door, 'B'/'Y'/'R' locked door (blue/yellow/red), 'E' exit pad.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "env"))
from doom_env.wad import Lump, write_wad  # noqa: E402

CELL = 64

WALL_TEX = ["STARTAN2", "STARTAN3", "BROWN1", "GRAY1", "STONE2", "STONE3", "BROWNGRN", "METAL",
            "TEKGREN2", "BRICK1", "CEMENT1", "COMPSPAN", "GSTONE1", "MARBLE1", "PIPE2", "SLADWALL"]
FLOOR_TEX = ["FLOOR0_1", "FLOOR0_3", "FLOOR4_8", "FLOOR5_1", "FLAT1", "FLAT5", "FLAT14", "GRASS1",
             "CEIL3_5", "FLOOR7_1", "RROCK09", "MFLR8_1"]
CEIL_TEX = ["CEIL1_1", "CEIL3_5", "CEIL5_1", "FLAT2", "TLITE6_4", "CEIL4_2"]

# keycard thing types and ZDoom lock numbers ("any key of that colour")
KEYS = {"B": (5, 130, "DOORBLU"), "Y": (6, 131, "DOORYEL"), "R": (13, 129, "DOORRED")}

MONSTERS = [  # (thing type, weight, min stage)
    (3004, 10, 0),  # zombieman
    (9, 7, 0),      # shotgun guy
    (3001, 9, 0),   # imp
    (3002, 5, 1),   # demon
    (58, 2, 2),     # spectre
    (3006, 2, 2),   # lost soul
    (65, 3, 2),     # chaingunner
    (3005, 2, 3),   # cacodemon
    (69, 1, 3),     # hell knight
]
ITEMS = [(2011, 6), (2012, 3), (2007, 5), (2008, 4), (2048, 2), (2049, 1), (2018, 2), (2019, 1),
         (2001, 2), (2002, 1), (2014, 4), (2015, 3)]


@dataclass
class LevelSpec:
    width: tuple[int, int] = (16, 24)
    height: tuple[int, int] = (16, 24)
    rooms: tuple[int, int] = (2, 4)
    room_size: tuple[int, int] = (3, 7)
    extra_edges: float = 0.2      # fraction of extra loop corridors
    door_prob: float = 0.3
    locks: int = 0                # 0, 1 or 2 keycard locks on the way to the exit
    monsters: tuple[int, int] = (0, 3)
    items: tuple[int, int] = (1, 4)
    stage: int = 0
    exit: bool = True
    arena: bool = False           # single big combat room
    tags: list[str] = field(default_factory=list)


def spec_for_stage(stage: int, rng: random.Random) -> LevelSpec:
    """Curriculum: stage 0 small/easy ... stage 4 full gauntlet. Draws a tier mix per stage."""
    s = max(0, min(stage, 4))
    kind = rng.choices(["combat", "explore", "keys", "gauntlet"],
                       weights=[[4, 4, 1, 0], [3, 4, 3, 1], [2, 3, 4, 2], [2, 3, 4, 3], [2, 3, 4, 4]][s])[0]
    if kind == "combat":
        size = 8 + 3 * s
        return LevelSpec(width=(size, size + 6), height=(size, size + 6), rooms=(1, 2), room_size=(5, 8 + s),
                         monsters=(1 + s, 2 + 3 * s), items=(2, 5), arena=True, stage=s, exit=rng.random() < 0.7,
                         tags=["combat"])
    if kind == "explore":
        size = 16 + 5 * s
        return LevelSpec(width=(size, size + 8), height=(size, size + 8), rooms=(3 + s, 5 + 2 * s),
                         monsters=(s, 2 + 2 * s), door_prob=0.3 + 0.1 * s, extra_edges=0.25, stage=s,
                         tags=["explore"])
    if kind == "keys":
        size = 18 + 4 * s
        return LevelSpec(width=(size, size + 8), height=(size, size + 8), rooms=(3 + s, 4 + 2 * s),
                         monsters=(s, 2 + 2 * s), locks=1 if s < 3 or rng.random() < 0.6 else 2,
                         door_prob=0.4, stage=s, tags=["keys"])
    size = 26 + 4 * s
    return LevelSpec(width=(size, size + 8), height=(size, size + 8), rooms=(5 + s, 7 + 2 * s),
                     monsters=(2 + s, 4 + 2 * s), items=(5 + s, 9 + 2 * s), locks=rng.choice([1, 2]), door_prob=0.5,
                     stage=s, tags=["gauntlet"])


# ----------------------------------------------------------------------------- layout
class Layout:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.g = [["#"] * w for _ in range(h)]
        self.region = [[-1] * w for _ in range(h)]  # room id, -2 corridor, -1 wall
        self.rooms: list[tuple[int, int, int, int]] = []  # r0, c0, r1, c1 (inclusive)
        self.things: list[tuple[int, int, int, int]] = []  # type, r, c, angle

    def inside(self, r, c):
        return 0 < r < self.h - 1 and 0 < c < self.w - 1

    def center(self, i):
        r0, c0, r1, c1 = self.rooms[i]
        return (r0 + r1) // 2, (c0 + c1) // 2

    def room_cells(self, i):
        r0, c0, r1, c1 = self.rooms[i]
        return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _place_rooms(L: Layout, spec: LevelSpec, rng: random.Random) -> None:
    want = rng.randint(*spec.rooms)
    lo, hi = spec.room_size
    for _ in range(400):
        if len(L.rooms) >= want:
            break
        rh, rw = rng.randint(lo, hi), rng.randint(lo, hi)
        if rh + 2 >= L.h or rw + 2 >= L.w:
            continue
        r0, c0 = rng.randint(1, L.h - rh - 1), rng.randint(1, L.w - rw - 1)
        r1, c1 = r0 + rh - 1, c0 + rw - 1
        if any(not (r1 + 2 < a0 or a1 + 2 < r0 or c1 + 2 < b0 or b1 + 2 < c0) for a0, b0, a1, b1 in L.rooms):
            continue
        i = len(L.rooms)
        L.rooms.append((r0, c0, r1, c1))
        for r, c in L.room_cells(i):
            L.g[r][c] = "."
            L.region[r][c] = i


def _carve_corridor(L: Layout, a: int, b: int, rng: random.Random) -> None:
    (r, c), (r2, c2) = L.center(a), L.center(b)
    horizontal_first = rng.random() < 0.5
    path = []
    if horizontal_first:
        path += [(r, cc) for cc in _span(c, c2)] + [(rr, c2) for rr in _span(r, r2)]
    else:
        path += [(rr, c) for rr in _span(r, r2)] + [(r2, cc) for cc in _span(c, c2)]
    for rr, cc in path:
        if L.g[rr][cc] == "#":
            L.g[rr][cc] = "."
            L.region[rr][cc] = -2


def _span(a, b):
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _mst(L: Layout) -> list[tuple[int, int]]:
    n = len(L.rooms)
    if n < 2:
        return []
    cen = [L.center(i) for i in range(n)]
    seen, edges, pq = {0}, [], []
    for j in range(1, n):
        heapq.heappush(pq, (abs(cen[0][0] - cen[j][0]) + abs(cen[0][1] - cen[j][1]), 0, j))
    while pq and len(seen) < n:
        _, i, j = heapq.heappop(pq)
        if j in seen:
            continue
        seen.add(j)
        edges.append((i, j))
        for k in range(n):
            if k not in seen:
                heapq.heappush(pq, (abs(cen[j][0] - cen[k][0]) + abs(cen[j][1] - cen[k][1]), j, k))
    return edges


def _bfs(L: Layout, start, blocked=frozenset()):
    dist = {start: 0}
    q = [start]
    for r, c in q:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if n not in dist and L.g[n[0]][n[1]] != "#" and n not in blocked:
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist


def _door_candidates(L: Layout):
    """Corridor cells that are a 1-wide chokepoint (walls on two opposite sides, floor on the others)."""
    out = []
    for r in range(1, L.h - 1):
        for c in range(1, L.w - 1):
            if L.g[r][c] != "." or L.region[r][c] != -2:
                continue
            ns = L.g[r - 1][c], L.g[r + 1][c]
            ew = L.g[r][c - 1], L.g[r][c + 1]
            if ns == ("#", "#") and "#" not in ew:
                out.append((r, c))
            elif ew == ("#", "#") and "#" not in ns:
                out.append((r, c))
    return out


def _separates(L: Layout, cell, a_cells, b_cells) -> bool:
    """True if blocking `cell` disconnects every cell of a_cells from b_cells."""
    d = _bfs(L, a_cells[0], blocked=frozenset([cell]))
    return not any(x in d for x in b_cells)


def generate(spec: LevelSpec, seed: int) -> Layout | None:
    rng = random.Random(seed)
    L = Layout(rng.randint(*spec.width), rng.randint(*spec.height))
    _place_rooms(L, spec, rng)
    if not L.rooms:
        return None
    n = len(L.rooms)
    edges = _mst(L)
    degree = [0] * n
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    start = 0
    # exit room: farthest MST leaf from start (keeps a single entrance so a lock really gates it)
    cen = [L.center(i) for i in range(n)]
    leaves = [i for i in range(n) if degree[i] == 1 and i != start]
    exit_room = max(leaves, key=lambda i: abs(cen[i][0] - cen[start][0]) + abs(cen[i][1] - cen[start][1])) if leaves else (n - 1 if n > 1 else 0)
    protected = {exit_room}
    key_room2 = None
    if spec.locks >= 2:
        others = [i for i in leaves if i not in (exit_room, start)]
        if others:
            key_room2 = rng.choice(others)
            protected.add(key_room2)
    for a, b in edges:
        _carve_corridor(L, a, b, rng)
    for _ in range(int(spec.extra_edges * n + rng.random())):
        a, b = rng.sample(range(n), 2) if n >= 2 else (0, 0)
        if a != b and a not in protected and b not in protected:
            _carve_corridor(L, a, b, rng)

    start_cells = L.room_cells(start)
    locks_needed = []
    if spec.locks >= 1 and exit_room != start:
        locks_needed.append(exit_room)
    if key_room2 is not None:
        locks_needed.append(key_room2)
    colors = rng.sample(list(KEYS), 3)
    lock_cells = {}
    for k, room in enumerate(locks_needed):
        cands = [cell for cell in _door_candidates(L) if _separates(L, cell, start_cells, L.room_cells(room))]
        if not cands:
            return None
        cell = rng.choice(cands)
        L.g[cell[0]][cell[1]] = colors[k]
        lock_cells[room] = (cell, colors[k])
    # plain doors
    def at_room_entrance(cell):
        r, c = cell
        return any(L.region[r + dr][c + dc] >= 0 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    for cell in _door_candidates(L):
        if at_room_entrance(cell) and rng.random() < spec.door_prob and all(abs(cell[0] - x[0][0]) + abs(cell[1] - x[0][1]) > 1 for x in lock_cells.values()):
            if L.g[cell[0]][cell[1]] == ".":
                # avoid adjacent doors
                if not any(L.g[cell[0] + dr][cell[1] + dc] in "DBYR" for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    L.g[cell[0]][cell[1]] = "D"

    # player start
    sr, sc = rng.choice(start_cells)
    L.things.append((1, sr, sc, rng.choice(range(0, 360, 45))))
    used = {(sr, sc)}

    # keys: key for exit lock goes in a room reachable without that lock (and without lock 2)
    reach_free = _bfs(L, (sr, sc), blocked=frozenset(c for c, _ in lock_cells.values()))
    if exit_room in lock_cells:
        col = lock_cells[exit_room][1]
        if key_room2 is not None and key_room2 in lock_cells:
            target_room = key_room2  # key 1 sits behind lock 2
        else:
            target_room = None
        if target_room is not None:
            cands = [x for x in L.room_cells(target_room) if x not in used]
        else:
            cands = [x for x in reach_free if L.region[x[0]][x[1]] >= 0 and L.region[x[0]][x[1]] not in (start, exit_room)
                     and L.g[x[0]][x[1]] == "."] or [x for x in reach_free if L.g[x[0]][x[1]] == "." and x not in used]
        kc = rng.choice(cands)
        L.things.append((KEYS[col][0], kc[0], kc[1], 0))
        used.add(kc)
    if key_room2 is not None and key_room2 in lock_cells:
        col = lock_cells[key_room2][1]
        cands = [x for x in reach_free if L.g[x[0]][x[1]] == "." and x not in used and L.region[x[0]][x[1]] != start] or \
                [x for x in reach_free if L.g[x[0]][x[1]] == "." and x not in used]
        kc = rng.choice(cands)
        L.things.append((KEYS[col][0], kc[0], kc[1], 0))
        used.add(kc)

    # exit pad in the exit room, far from its entrance
    if spec.exit:
        cells = [x for x in L.room_cells(exit_room) if x not in used]
        if exit_room == start:
            cells = [x for x in cells if abs(x[0] - sr) + abs(x[1] - sc) >= 3] or cells
        er, ec = rng.choice(cells)
        L.g[er][ec] = "E"
        used.add((er, ec))

    floor_cells = [(r, c) for r in range(L.h) for c in range(L.w) if L.g[r][c] == "." and (r, c) not in used]
    far = [x for x in floor_cells if abs(x[0] - sr) + abs(x[1] - sc) > 4]
    mons = [m for m in MONSTERS if m[2] <= spec.stage]
    for _ in range(rng.randint(*spec.monsters)):
        if not far:
            break
        cell = far.pop(rng.randrange(len(far)))
        t = rng.choices([m[0] for m in mons], weights=[m[1] for m in mons])[0]
        L.things.append((t, cell[0], cell[1], rng.choice(range(0, 360, 45))))
        used.add(cell)
    free = [x for x in floor_cells if x not in used]
    for _ in range(rng.randint(*spec.items)):
        if not free:
            break
        cell = free.pop(rng.randrange(len(free)))
        t = rng.choices([i[0] for i in ITEMS], weights=[i[1] for i in ITEMS])[0]
        L.things.append((t, cell[0], cell[1], 0))
    return L


# ----------------------------------------------------------------------------- UDMF writer
def to_udmf(L: Layout, seed: int) -> str:
    rng = random.Random(seed ^ 0x5EED)
    sectors: list[str] = []
    sector_of: dict = {}

    def add_sector(floor, ceil, light, ftex, ctex):
        sectors.append(f'sector {{ texturefloor = "{ftex}"; textureceiling = "{ctex}"; heightfloor = {floor}; '
                       f'heightceiling = {ceil}; lightlevel = {light}; }}')
        return len(sectors) - 1

    room_style = []
    for i in range(len(L.rooms)):
        floor = rng.choice([0, 0, 0, 8, 16, -8])
        ceil = floor + rng.choice([128, 128, 160, 192, 256])
        light = rng.choice([144, 160, 176, 192, 208, 224, 255])
        sec = add_sector(floor, ceil, light, rng.choice(FLOOR_TEX), rng.choice(CEIL_TEX))
        room_style.append((sec, rng.choice(WALL_TEX), floor, ceil))
    corr_tex = rng.choice(WALL_TEX)
    corr_sec = add_sector(0, rng.choice([112, 128]), rng.choice([128, 160, 192]), rng.choice(FLOOR_TEX), rng.choice(CEIL_TEX))
    exit_sec = None
    for r in range(L.h):
        for c in range(L.w):
            ch = L.g[r][c]
            if ch == "#":
                continue
            if ch in "DBYR":
                sector_of[(r, c)] = add_sector(0, 0, 160, "FLAT5_4", "CEIL1_1")
            elif ch == "E":
                reg = L.region[r][c]
                fl, ce = (room_style[reg][2], room_style[reg][3]) if reg >= 0 else (0, 128)
                exit_sec = add_sector(fl, ce, 255, "GATE3", "CEIL1_1")
                sector_of[(r, c)] = exit_sec
            elif L.region[r][c] >= 0:
                sector_of[(r, c)] = room_style[L.region[r][c]][0]
            else:
                sector_of[(r, c)] = corr_sec

    def wall_tex(cell):
        reg = L.region[cell[0]][cell[1]]
        return room_style[reg][1] if reg >= 0 else corr_tex

    verts: dict[tuple[int, int], int] = {}
    vlist: list[str] = []

    def v(x, y):
        if (x, y) not in verts:
            verts[(x, y)] = len(vlist)
            vlist.append(f"vertex {{ x = {float(x)}; y = {float(y)}; }}")
        return verts[(x, y)]

    sides: list[str] = []
    lines: list[str] = []

    def side(sec, top="-", mid="-", bot="-"):
        sides.append(f'sidedef {{ sector = {sec}; texturetop = "{top}"; texturemiddle = "{mid}"; texturebottom = "{bot}"; }}')
        return len(sides) - 1

    def cell_xy(r, c):  # lower-left corner of cell; row 0 at top
        return c * CELL, -(r + 1) * CELL

    def oriented(p, q, front_cell):
        fx, fy = cell_xy(*front_cell)
        fx += CELL / 2
        fy += CELL / 2
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        dx, dy = q[0] - p[0], q[1] - p[1]
        cross = dx * (fy - my) - dy * (fx - mx)
        return (p, q) if cross < 0 else (q, p)

    def edge(a, b, p, q):
        ca, cb = L.g[a[0]][a[1]], L.g[b[0]][b[1]]
        if ca == "#" and cb == "#":
            return
        if ca == "#" or cb == "#":
            f = b if ca == "#" else a
            p2, q2 = oriented(p, q, f)
            tex = "DOORTRAK" if L.g[f[0]][f[1]] in "DBYR" else wall_tex(f)
            s = side(sector_of[f], mid=tex)
            lines.append(f"linedef {{ v1 = {v(*p2)}; v2 = {v(*q2)}; sidefront = {s}; blocking = true; dontpegbottom = true; }}")
            return
        sa, sb = sector_of[a], sector_of[b]
        if sa == sb:
            return
        # orient so the "outer" cell is in front (door/exit cell behind)
        if ca in "DBYRE" and cb not in "DBYRE":
            a, b, ca, cb, sa, sb = b, a, cb, ca, sb, sa
        p2, q2 = oriented(p, q, a)
        wt = wall_tex(a)
        if cb in "DBYR":
            top = "BIGDOOR2" if cb == "D" else KEYS[cb][2]
            s1, s2 = side(sa, top=top), side(sb)
            if cb == "D":
                spec = "special = 12; arg0 = 0; arg1 = 16; arg2 = 150; playeruse = true; monsteruse = true; repeatspecial = true;"
            else:
                spec = (f"special = 13; arg0 = 0; arg1 = 16; arg2 = 0; arg3 = {KEYS[cb][1]}; playeruse = true; "
                        "monsteruse = false; repeatspecial = true;")
            lines.append(f"linedef {{ v1 = {v(*p2)}; v2 = {v(*q2)}; sidefront = {s1}; sideback = {s2}; twosided = true; {spec} }}")
        elif cb == "E":
            s1, s2 = side(sa, top=wt, bot=wt), side(sb, top=wt, bot=wt)
            lines.append(f"linedef {{ v1 = {v(*p2)}; v2 = {v(*q2)}; sidefront = {s1}; sideback = {s2}; twosided = true; special = 243; playercross = true; }}")
        else:
            s1, s2 = side(sa, top=wt, bot=wt), side(sb, top=wall_tex(b), bot=wall_tex(b))
            lines.append(f"linedef {{ v1 = {v(*p2)}; v2 = {v(*q2)}; sidefront = {s1}; sideback = {s2}; twosided = true; }}")

    for r in range(L.h):
        for c in range(L.w):
            x, y = cell_xy(r, c)
            if c + 1 < L.w:  # east edge
                edge((r, c), (r, c + 1), (x + CELL, y), (x + CELL, y + CELL))
            if r + 1 < L.h:  # south edge (row r+1 is below)
                edge((r, c), (r + 1, c), (x, y), (x + CELL, y))

    things = []
    jit = random.Random(seed ^ 0x7417)
    for t, r, c, ang in L.things:
        x, y = cell_xy(r, c)
        j = 0 if t == 1 else 12
        x += CELL / 2 + jit.uniform(-j, j)
        y += CELL / 2 + jit.uniform(-j, j)
        things.append(f"thing {{ x = {x:.1f}; y = {y:.1f}; angle = {ang}; type = {t}; skill1 = true; skill2 = true; "
                      "skill3 = true; skill4 = true; skill5 = true; single = true; }")
    return "\n".join(['namespace = "zdoom";'] + vlist + sectors + sides + lines + things) + "\n"


def write_level(path: str | Path, spec: LevelSpec, seed: int, tries: int = 20) -> Layout:
    """Generate a valid level for `spec` and write it as a one-map PWAD (MAP01). Returns the layout."""
    for k in range(tries):
        L = generate(spec, seed * 7919 + k)
        if L is not None:
            write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", to_udmf(L, seed).encode()), Lump("ENDMAP", b"")])
            return L
    raise RuntimeError(f"could not generate level for {spec}")


def ascii_map(L: Layout) -> str:
    g = [row[:] for row in L.g]
    sym = {1: "P", 5: "b", 6: "y", 13: "r"}
    for t, r, c, _ in L.things:
        if g[r][c] in ".":
            g[r][c] = sym.get(t, "m" if any(t == m[0] for m in MONSTERS) else "+")
    return "\n".join("".join(row) for row in g)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gen.wad")
    a = ap.parse_args()
    spec = spec_for_stage(a.stage, random.Random(a.seed))
    L = write_level(a.out, spec, a.seed)
    print(spec.tags, spec.locks)
    print(ascii_map(L))
