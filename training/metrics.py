"""Per-episode gameplay metrics for the benchmark (scoring side only, never seen by the agent).

Adds what the reward shaper does not count: aim accuracy, damage dealt, weapon switches, secrets,
kill percentage and rooms visited / cleared. Reads the engine directly, so the env stays untouched.

A "room" is a generated room rectangle when the layout is known, otherwise a 512-unit block of the
map that held monsters at the start. A room is cleared once every monster that started in it is dead.
"""

from __future__ import annotations

import math

import vizdoom as vd

GV = vd.GameVariable
AMMO = (GV.AMMO2, GV.AMMO3, GV.AMMO5, GV.AMMO6)  # bullets, shells, rockets, cells
CELL = 64.0
BLOCK = 512.0


def _alive(o) -> bool:
    return o.category == "Monster" and not o.name.startswith("Dead")


class EpisodeMetrics:
    def __init__(self, env, layout=None):
        self.g = env._game
        g = self.g
        st = g.get_state()
        mons = {o.id: (o.position_x, o.position_y) for o in st.objects if _alive(o)}
        self.total = len(mons)
        self.alive = set(mons)
        if layout is not None:  # rooms from the generator: (r0, c0, r1, c1) in cells, row 0 on top
            self.rects = [(c0 * CELL, -(r1 + 1) * CELL, (c1 + 1) * CELL, -r0 * CELL) for r0, c0, r1, c1 in layout.rooms]
            self.room_of = lambda x, y: next((i for i, (x0, y0, x1, y1) in enumerate(self.rects)
                                              if x0 <= x <= x1 and y0 <= y <= y1), None)
            self.n_rooms = len(self.rects)
        else:
            self.room_of = lambda x, y: (math.floor(x / BLOCK), math.floor(y / BLOCK))
            self.n_rooms = None
        self.room_mons: dict = {}
        for i, (x, y) in mons.items():
            r = self.room_of(x, y)
            if r is not None:
                self.room_mons.setdefault(r, set()).add(i)
        self.visited = set()
        self.weapon = g.get_game_variable(GV.SELECTED_WEAPON)
        self.weapons = {int(self.weapon)}
        self.ammo = [g.get_game_variable(a) for a in AMMO]
        self.hit = g.get_game_variable(GV.HITCOUNT)
        self.switches = self.shots = self.hits = self.fire_steps = 0
        # movement quality
        self.steps = self.flips = self.idle = self.contact = self.dither = 0
        self.dist = 0.0
        self.last_turn = 0     # -1 right, +1 left: the last turn direction pressed
        self.last_pos = (g.get_game_variable(GV.POSITION_X), g.get_game_variable(GV.POSITION_Y))
        self.cells128 = {(math.floor(self.last_pos[0] / 128), math.floor(self.last_pos[1] / 128))}
        self.hist = []
        self._visit()

    def _visit(self):
        r = self.room_of(self.g.get_game_variable(GV.POSITION_X), self.g.get_game_variable(GV.POSITION_Y))
        if r is not None:
            self.visited.add(r)

    def step(self, action) -> None:
        g = self.g
        self.fire_steps += int(action[6])
        self.steps += 1
        turn = int(action[4]) - int(action[5])
        if turn and self.last_turn and turn != self.last_turn:
            self.flips += 1  # left then right (or back): wobbling instead of holding a line
        if turn:
            self.last_turn = turn
        if not (action[0] or action[1] or action[2] or action[3]):
            self.idle += 1
        x, y = g.get_game_variable(GV.POSITION_X), g.get_game_variable(GV.POSITION_Y)
        self.dist += math.hypot(x - self.last_pos[0], y - self.last_pos[1])
        self.last_pos = (x, y)
        self.cells128.add((math.floor(x / 128), math.floor(y / 128)))
        self.hist.append((x, y))
        if len(self.hist) >= 50 and len(self.hist) % 50 == 0:  # 50 steps that went nowhere new
            px, py = self.hist[-50]
            if math.hypot(x - px, y - py) < 96:
                self.dither += 1
        w = g.get_game_variable(GV.SELECTED_WEAPON)
        if w != self.weapon:
            self.switches += 1
            self.weapons.add(int(w))
            self.weapon = w
        ammo = [g.get_game_variable(a) for a in AMMO]
        fired = any(b < a for a, b in zip(self.ammo, ammo))
        self.ammo = ammo
        hit = g.get_game_variable(GV.HITCOUNT)
        if fired:
            self.shots += 1
            self.hits += int(hit > self.hit)
        self.hit = hit
        if not g.is_episode_finished():
            self._visit()
            self.alive = {o.id for o in g.get_state().objects if _alive(o)}

    def result(self, kills: int) -> dict:
        g = self.g
        hostile = self.room_mons
        cleared = sum(1 for ids in hostile.values() if not (ids & self.alive))
        # the engine has no object list after the last tic: trust the kill counter for a full clear
        if self.total and kills >= self.total:
            cleared = len(hostile)
        return dict(
            monsters=self.total, kill_pct=kills / self.total if self.total else float("nan"),
            shots=self.shots, hits=self.hits, acc=self.hits / self.shots if self.shots else float("nan"),
            dmg_dealt=g.get_game_variable(GV.DAMAGECOUNT), fire_steps=self.fire_steps,
            wpn_switches=self.switches, wpn_used=len(self.weapons), secrets=int(g.get_game_variable(GV.SECRETCOUNT)),
            rooms_seen=len(self.visited), rooms_total=self.n_rooms if self.n_rooms is not None else float("nan"),
            rooms_hostile=len(hostile), rooms_cleared=cleared,
            steps=self.steps, turn_flips=self.flips, idle_steps=self.idle, dist=self.dist,
            new_cells=len(self.cells128), stuck_windows=self.dither,
        )
