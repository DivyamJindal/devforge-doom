"""Procedural Doom level generator: random dungeon layouts -> UDMF (zdoom namespace) PWADs that ViZDoom loads directly
(ZDoom builds nodes itself), using the same conventions as the official test fixtures.

  python training/mapgen.py --out /tmp/maps --n 8 --tier 3       # writes /tmp/maps/gen_*.wad and prints an ASCII preview

Tiers mirror the hidden evaluation suite (tier 0 is an easy on-ramp for the curriculum: small, few monsters):
  1 combat     a few connected arenas, many monsters, health/ammo, exit
  2 explore    6-12 rooms, corridors, doors, dead ends, exit far from the start
  3 progress   like 2 plus 1-2 key -> locked door chains guarding the exit (keys in side branches)
  4 gauntlet   3 + more monsters, scarcer items
Grid legend: '#' wall, '.' floor, 'D' door, 'r/b/y' key, 'R/B/Y' locked door, 'E' exit pad, 'P' start,
'm' monster, 'h' health, 'a' ammo, 'w' weapon, 'A' armor.
"""
from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

CELL = 64
KEY_THING = {"r": 13, "b": 5, "y": 6}          # red / blue / yellow keycard
LOCK_NUM = {"R": 1, "B": 2, "Y": 3}             # ZDoom lock numbers (card or skull of that colour)
MONSTERS = [(3004, 5), (9, 3), (3001, 4), (3002, 2), (58, 1), (65, 1), (3005, 0.6), (3006, 0.6)]
ITEMS = {"h": [(2011, 3), (2012, 1)], "a": [(2007, 3), (2008, 2), (2048, 1), (2049, 1)],
         "w": [(2001, 3), (2002, 1)], "A": [(2018, 2), (2015, 3)]}
FLOOR_TEX = ["FLOOR0_1", "FLOOR4_8", "FLAT5_4", "FLOOR5_1", "CEIL5_1", "FLAT1"]
WALL_TEX = ["STARTAN2", "BROWN1", "GRAY1", "STONE2", "BRICK1", "METAL1"]


def _wchoice(rng, pairs):
    tot = sum(w for _, w in pairs); x = rng.random() * tot
    for v, w in pairs:
        x -= w
        if x <= 0:
            return v
    return pairs[-1][0]


# ----------------------------------------------------------------------------------- layout
def layout(rng: random.Random, tier: int) -> list[list[str]]:
    big = tier >= 2
    W, H = (rng.randint(30, 46), rng.randint(26, 40)) if big else (rng.randint(20, 30), rng.randint(16, 26))
    g = [["#"] * W for _ in range(H)]
    n_target = rng.randint(6, 12) if big else rng.randint(2, 4)
    rooms = []
    for _ in range(400):
        if len(rooms) >= n_target:
            break
        w = rng.randint(3, 7) if big else rng.randint(5, 10)
        h = rng.randint(3, 7) if big else rng.randint(4, 9)
        x, y = rng.randint(1, W - w - 2), rng.randint(1, H - h - 2)
        if any(x - 2 < rx + rw and rx - 2 < x + w and y - 2 < ry + rh and ry - 2 < y + h for rx, ry, rw, rh in rooms):
            continue
        rooms.append((x, y, w, h))
    for x, y, w, h in rooms:
        for j in range(y, y + h):
            for i in range(x, x + w):
                g[j][i] = "."
    cen = [(x + w // 2, y + h // 2) for x, y, w, h in rooms]
    # minimum spanning tree over room centres + a few loops
    edges = sorted(((abs(a[0] - b[0]) + abs(a[1] - b[1]), i, k) for i, a in enumerate(cen) for k, b in enumerate(cen) if i < k))
    parent = list(range(len(rooms)))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]; u = parent[u]
        return u
    links = []
    for _, i, k in edges:
        if find(i) != find(k):
            parent[find(i)] = find(k); links.append((i, k))
    extra = [e for e in edges if (e[1], e[2]) not in links]
    for _, i, k in extra[: rng.randint(0, 2 if big else 1)]:
        links.append((i, k))
    for i, k in links:
        (x0, y0), (x1, y1) = cen[i], cen[k]
        wide = (not big) or rng.random() < 0.25
        path = ([(x, y0) for x in range(min(x0, x1), max(x0, x1) + 1)] + [(x1, y) for y in range(min(y0, y1), max(y0, y1) + 1)]
                if rng.random() < 0.5 else
                [(x0, y) for y in range(min(y0, y1), max(y0, y1) + 1)] + [(x, y1) for x in range(min(x0, x1), max(x0, x1) + 1)])
        for x, y in path:
            g[y][x] = "."
            if wide and 0 < y + 1 < H - 1:
                g[y + 1][x] = "."
    return g, rooms


def _is_room(rooms, x, y):
    return any(rx <= x < rx + rw and ry <= y < ry + rh for rx, ry, rw, rh in rooms)


def _door_spots(g, rooms):
    """1-wide corridor cells (walls on both sides) adjacent to a room: natural door positions."""
    H, W = len(g), len(g[0]); out = []
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if g[y][x] != "." or _is_room(rooms, x, y):
                continue
            ns = g[y - 1][x] == "#" and g[y + 1][x] == "#" and g[y][x - 1] != "#" and g[y][x + 1] != "#"
            ew = g[y][x - 1] == "#" and g[y][x + 1] == "#" and g[y - 1][x] != "#" and g[y + 1][x] != "#"
            near = any(_is_room(rooms, x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            if (ns or ew) and near:
                out.append((x, y))
    return out


def _bfs(g, start, passable):
    H, W = len(g), len(g[0]); dist = {start: 0}; q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if 0 <= n[0] < W and 0 <= n[1] < H and n not in dist and passable(g[n[1]][n[0]]):
                dist[n] = dist[(x, y)] + 1; q.append(n)
    return dist


def populate(rng: random.Random, g, rooms, tier: int) -> list[list[str]]:
    H, W = len(g), len(g[0])
    start_room = rng.randrange(len(rooms))
    sx, sy, sw, sh = rooms[start_room]
    start = (sx + sw // 2, sy + sh // 2)
    open_ = lambda c: c != "#"
    dist = _bfs(g, start, open_)
    # exit: a cell in the farthest room (by path length), against a wall
    far = max(range(len(rooms)), key=lambda r: dist.get((rooms[r][0] + rooms[r][2] // 2, rooms[r][1] + rooms[r][3] // 2), -1))
    if far == start_room and len(rooms) > 1:
        far = (start_room + 1) % len(rooms)
    ex, ey, ew, eh = rooms[far]
    exit_cell = (ex + rng.randrange(ew), ey + rng.randrange(eh))
    if exit_cell == start:
        exit_cell = (ex, ey)
    # doors
    spots = _door_spots(g, rooms)
    rng.shuffle(spots)
    n_doors = 0 if tier == 1 else min(len(spots), rng.randint(0, 2) if tier == 0 else rng.randint(2, 6))
    doors = spots[:n_doors]
    for x, y in doors:
        g[y][x] = "D"
    g[exit_cell[1]][exit_cell[0]] = "E"
    # locks (tier >= 3): lock a door on the start->exit path so that the exit side is cut off; key on start side
    if tier >= 3:
        colours = rng.sample(["r", "b", "y"], k=rng.choice([1, 1, 2]))
        for col in colours:
            reach = _bfs(g, start, lambda c: c not in "#RBY")
            if exit_cell not in reach:
                break
            # walk back from exit to find doors on the path; pick the one closest to the exit that disconnects it
            cand = [d for d in doors if g[d[1]][d[0]] == "D" and d in reach]
            rng.shuffle(cand)
            done = False
            for d in sorted(cand, key=lambda d: -reach[d]):
                g[d[1]][d[0]] = col.upper()
                side = _bfs(g, start, lambda c: c not in "#RBY")
                if exit_cell not in side and len(side) > 12:
                    floor = [c for c in side if g[c[1]][c[0]] == "." and c != start and _is_room(rooms, *c)]
                    if not floor:
                        floor = [c for c in side if g[c[1]][c[0]] == "." and c != start]
                    if floor:
                        kx, ky = max(floor, key=lambda c: side[c] + rng.random() * 6)    # far from start: forces exploring
                        g[ky][kx] = col; done = True; break
                g[d[1]][d[0]] = "D"
            if not done:
                break
    # monsters and items
    floor = [(x, y) for y in range(H) for x in range(W) if g[y][x] == "." and abs(x - start[0]) + abs(y - start[1]) > 4]
    rng.shuffle(floor)
    n_cells = len(floor)
    density = {0: 0.012, 1: 0.06, 2: 0.025, 3: 0.025, 4: 0.05}[tier]
    n_mon = max(1, int(n_cells * density * rng.uniform(0.6, 1.4)))
    item_mul = 0.5 if tier == 4 else 1.0
    n_items = {"h": int(n_cells * 0.012 * item_mul) + 1, "a": int(n_cells * 0.015 * item_mul) + 1,
               "w": rng.choice([0, 1, 1]), "A": rng.choice([0, 1])}
    for _ in range(min(n_mon, len(floor))):
        x, y = floor.pop(); g[y][x] = "m"
    for k, n in n_items.items():
        for _ in range(min(n, len(floor))):
            x, y = floor.pop(); g[y][x] = k
    g[start[1]][start[0]] = "P"
    return g


# ----------------------------------------------------------------------------------- UDMF writer
def to_udmf(g, rng: random.Random) -> str:
    H, W = len(g), len(g[0])
    floor_tex, wall_tex = rng.choice(FLOOR_TEX), rng.choice(WALL_TEX)
    ceil_h = rng.choice([128, 128, 160, 192])
    light = rng.choice([160, 192, 192, 224])
    sectors = [f'sector {{ texturefloor = "{floor_tex}"; textureceiling = "CEIL1_1"; heightfloor = 0; heightceiling = {ceil_h}; lightlevel = {light}; }}']
    sec_of = {}
    for y in range(H):
        for x in range(W):
            c = g[y][x]
            if c == "#":
                continue
            if c in "DRBYE":                                   # every door / exit cell gets its own sector
                sec_of[(x, y)] = len(sectors)
                tex = {"E": "GATE3"}.get(c, "FLAT5_4")
                hc = ceil_h if c == "E" else 0
                sectors.append(f'sector {{ texturefloor = "{tex}"; textureceiling = "CEIL1_1"; heightfloor = 0; heightceiling = {hc}; lightlevel = {light}; }}')
            else:
                sec_of[(x, y)] = 0
    verts, vid, sides, lines = [], {}, [], []

    def V(px, py):
        if (px, py) not in vid:
            vid[(px, py)] = len(verts); verts.append(f"vertex {{ x = {px * CELL:.1f}; y = {py * CELL:.1f}; }}")
        return vid[(px, py)]

    def side(sec, mid):
        sides.append(f'sidedef {{ sector = {sec}; texturetop = "{wall_tex}"; texturemiddle = "{mid}"; texturebottom = "-"; }}')
        return len(sides) - 1

    def kind(c):
        return "door" if c in "DRBY" else "exit" if c == "E" else "floor"

    # map row y of the grid to world y = (H - 1 - y) so the ASCII preview is north-up
    for y in range(H):
        for x in range(W):
            if (x, y) not in sec_of:
                continue
            a = sec_of[(x, y)]; c = g[y][x]; wy = H - 1 - y
            # edges: (neighbour, v1, v2) with v1->v2 oriented so THIS cell is on the right (front) side
            for (nx, ny), p1, p2 in (((x, y - 1), (x, wy + 1), (x + 1, wy + 1)),     # north edge
                                     ((x + 1, y), (x + 1, wy + 1), (x + 1, wy)),     # east
                                     ((x, y + 1), (x + 1, wy), (x, wy)),             # south
                                     ((x - 1, y), (x, wy), (x, wy + 1))):            # west
                if (nx, ny) not in sec_of:
                    s = side(a, wall_tex)
                    lines.append(f"linedef {{ v1 = {V(*p1)}; v2 = {V(*p2)}; sidefront = {s}; blocking = true; dontpegbottom = true; }}")
                    continue
                b = sec_of[(nx, ny)]
                if a == b:
                    continue
                nc = g[ny][nx]
                # emit each two-sided boundary once, from the non-door / non-exit side so specials face the player
                if kind(c) != "floor" and kind(nc) == "floor":
                    continue
                if kind(c) == kind(nc) and (x, y) > (nx, ny):
                    continue
                s1, s2 = side(a, "-"), side(b, "-")
                head = f"linedef {{ v1 = {V(*p1)}; v2 = {V(*p2)}; sidefront = {s1}; sideback = {s2}; twosided = true;"
                if nc in "DRBY":
                    if nc == "D":
                        lines.append(f"{head} special = 12; arg0 = 0; arg1 = 16; arg2 = 150; playeruse = true; monsteruse = false; repeatspecial = true; }}")
                    else:
                        lines.append(f"{head} special = 13; arg0 = 0; arg1 = 16; arg2 = 150; arg3 = {LOCK_NUM[nc]}; playeruse = true; repeatspecial = true; }}")
                elif nc == "E":
                    lines.append(f"{head} special = 243; playercross = true; }}")
                else:
                    lines.append(f"{head} }}")
    things = []
    for y in range(H):
        for x in range(W):
            c = g[y][x]; px, py = (x + 0.5) * CELL, (H - 1 - y + 0.5) * CELL
            t = None
            if c == "P":
                t = 1
            elif c == "m":
                t = _wchoice(rng, MONSTERS)
            elif c in ITEMS:
                t = _wchoice(rng, ITEMS[c])
            elif c in KEY_THING:
                t = KEY_THING[c]
            if t is not None:
                ang = rng.choice([0, 90, 180, 270])
                things.append(f"thing {{ x = {px:.1f}; y = {py:.1f}; angle = {ang}; type = {t}; skill1 = true; skill2 = true; "
                              f"skill3 = true; skill4 = true; skill5 = true; single = true; }}")
    return "\n".join(['namespace = "zdoom";', *verts, *sectors, *sides, *lines, *things]) + "\n"


def write_wad(path: str | Path, textmap: str) -> None:
    import struct
    lumps = [("MAP01", b""), ("TEXTMAP", textmap.encode()), ("ENDMAP", b"")]
    data = b"".join(d for _, d in lumps)
    off = 12; dirs = b""
    for name, d in lumps:
        dirs += struct.pack("<ii8s", off, len(d), name.encode().ljust(8, b"\0")); off += len(d)
    Path(path).write_bytes(b"PWAD" + struct.pack("<ii", len(lumps), 12 + len(data)) + data + dirs)


def generate(seed: int, tier: int, path: str | Path) -> list[str]:
    """Write one level to `path` (MAP01). Returns the ASCII grid (for previews and curriculum stats)."""
    rng = random.Random(seed * 7919 + tier)
    for attempt in range(20):
        g, rooms = layout(rng, tier)
        if len(rooms) < (2 if tier <= 1 else 4):
            continue
        g = populate(rng, g, rooms, tier)
        has_key = any(c in "rby" for row in g for c in row)
        if any("E" in row for row in g) and (tier < 3 or has_key or attempt >= 15):
            break
    write_wad(path, to_udmf(g, rng))
    return ["".join(r) for r in g]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/maps"); ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tier", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        rows = generate(a.seed + i, a.tier, Path(a.out) / f"gen_t{a.tier}_{a.seed + i}.wad")
        if i == 0:
            print("\n".join(rows))
    print(f"wrote {a.n} maps to {a.out}")
