"""Multiprocess vectorised DOOM environments with procedural maps, curriculum and feature extraction.

Each worker process owns `envs_per_worker` DoomEnv instances. Workers generate fresh levels
(training/mapgen.py) for the current curriculum stage, run the agent's SpatialMemory (same code as
evaluation), shape rewards from privileged state and send back ready-to-batch feature arrays.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "training"))

FEAT_KEYS = ("rays", "map", "vec", "ent", "ent_type", "ent_mask")


def _hires_timer():
    if os.name == "nt":  # Windows sleeps in 15.6 ms quanta by default: ViZDoom crawls without this
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)


def _fixture_lumps():
    """Hand-made maps mixed into every generated WAD: the template test map + its 13 fixtures."""
    from doom_env.wad import read_wad

    out = []
    for f in [ROOT / "maps" / "two_rooms.wad", *sorted((ROOT / "tests" / "fixtures").glob("*.wad"))]:
        lumps = read_wad(f)
        tm = [l for l in lumps if l.name == "TEXTMAP"]
        if tm:
            out.append(tm[0].data)
    return out


def _make_env_retry(wad, name, tries=6):
    """ViZDoom instances started at the same instant can race on shared files; retry with jitter."""
    import time

    from doom_env import make_env

    for k in range(tries):
        try:
            return make_env(wad, name, render_buffers=False)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(0.5 + random.random() * 2.0)


MAPS_PER_WAD = 32
GEN_PER_WAD = 29  # MAP01..MAP29 generated, MAP30..MAP32 hand-made fixtures


class _Slot:
    """One environment. Levels live in a 32-map PWAD; switching maps inside the running engine costs
    ~30 ms, while creating a new DoomGame costs ~1 s, so a fresh WAD is only built every few dozen
    episodes (or when the curriculum stage changes)."""

    def __init__(self, wid, idx, ctl, seed):
        self.idx = idx
        self.stage_val, self.beta_val, self.mode_val = ctl
        self.rng = random.Random(seed)
        self.env = None
        self.fd_env = None  # Freedoom 2 campaign env (IWAD maps), created lazily
        self.wad_no = 0
        self.wad_path = None
        self.wad_stage = -1
        self.wad_tags = {}
        self.wad_layouts = {}
        self.layout = None
        self.teacher = None
        self.queue = []
        self.fixtures = _fixture_lumps()
        self.tags = []
        self.cur = None

    def _build_wad(self, stage):
        from doom_env import make_env
        from doom_env.wad import Lump, write_wad
        from mapgen import generate, spec_for_stage, to_udmf

        lumps, tags, layouts = [], {}, {}
        for i in range(GEN_PER_WAD):
            spec = spec_for_stage(stage, self.rng)
            L = None
            while L is None:
                sd = self.rng.randrange(1 << 30)
                L = generate(spec, sd)
            name = f"MAP{i + 1:02d}"
            lumps += [Lump(name, b""), Lump("TEXTMAP", to_udmf(L, sd).encode()), Lump("ENDMAP", b"")]
            tags[name] = spec.tags + ([] if spec.exit else ["noexit"])
            layouts[name] = L
        for j, tm in enumerate(self.rng.sample(self.fixtures, MAPS_PER_WAD - GEN_PER_WAD)):
            name = f"MAP{GEN_PER_WAD + j + 1:02d}"
            lumps += [Lump(name, b""), Lump("TEXTMAP", tm), Lump("ENDMAP", b"")]
            tags[name] = ["fixture"]
        old = self.wad_path
        self.wad_no += 1
        self.wad_path = Path(tempfile.gettempdir()) / f"doomgen_{os.getpid()}_{self.idx}_{self.wad_no}.wad"
        write_wad(self.wad_path, lumps)
        if self.env is not None:
            self.env.close()
        self.env = _make_env_retry(self.wad_path, "MAP01")
        if old is not None:
            try:
                old.unlink()
            except OSError:
                pass
        self.wad_stage = stage
        self.wad_tags = tags
        self.wad_layouts = layouts
        self.queue = [f"MAP{i:02d}" for i in range(1, MAPS_PER_WAD + 1)]
        self.rng.shuffle(self.queue)
        self.queue = [m for m in self.queue for _ in range(2)]  # two episodes per level

    @staticmethod
    def _switch(env, name, ticks):
        from doom_env.mapinfo import load_mapinfo

        if env.map_name != name:
            env.map_name = name
            env.mapinfo = load_mapinfo(env.wad, name)
            env._geometry = None
            env._sensors = None
            env._game.set_doom_map(name)
        env.episode_ticks = ticks
        env._game.set_episode_timeout(ticks)

    def _pick(self):
        stage = int(self.stage_val.value)
        ticks = self.rng.choice([2100, 2800, 3500, 4200])
        distill = int(self.mode_val.value) == 2
        self.layout = None
        if not distill and self.rng.random() < 0.03 * stage:
            import vizdoom as vd
            from doom_env import make_env

            if self.fd_env is None:
                iwad = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
                self.fd_env = _make_env_retry(iwad, "MAP01")
            self._switch(self.fd_env, f"MAP{self.rng.randint(1, 32):02d}", 4200)
            self.tags = ["freedoom"]
            return self.fd_env
        if not self.queue or stage != self.wad_stage:
            self._build_wad(stage)
        name = self.queue.pop()
        while distill and name not in self.wad_layouts and self.queue:  # teacher needs a layout
            name = self.queue.pop()
        if name not in self.wad_layouts and distill:
            self._build_wad(stage)
            name = self.queue.pop()
        self._switch(self.env, name, ticks)
        self.tags = self.wad_tags[name]
        self.layout = self.wad_layouts.get(name)
        return self.env

    def reset(self):
        from rewards import RewardShaper
        from spatial_memory import SpatialMemory

        self.cur = self._pick()
        obs, _ = self.cur.reset(seed=self.rng.randrange(1 << 30))
        self.mem = SpatialMemory()
        self.shaper = RewardShaper()
        self.shaper.reset(self.cur.privileged_state())
        self.ret = 0.0
        self.len = 0
        self.obs = obs
        self.teacher = None
        if self.layout is not None and int(self.mode_val.value) >= 1:
            from teacher import Teacher

            self.teacher = Teacher(self.layout, seed=self.rng.randrange(1 << 30))
        return obs, self.mem.update(obs, None)

    def step(self, choice):
        """choice: the learner's head choices (6,). In distill mode the teacher's action is executed
        with probability beta (DAgger). Returns the teacher's label for the state (or -1s)."""
        from model import to_buttons
        from rewards import aux_targets
        from teacher import buttons_to_choice

        buttons = to_buttons(np.asarray(choice)[None])[0]
        label = np.full(6, -1, np.int64)
        if self.teacher is not None:
            tb = self.teacher.act(self.obs, self.cur.privileged_state())
            label = buttons_to_choice(tb)
            if self.rng.random() < float(self.beta_val.value):
                buttons = tb
        obs, _, term, trunc, info = self.cur.step(buttons)
        ps = self.cur.privileged_state()
        r = self.shaper(ps, buttons)
        self.ret += r
        self.len += 1
        done = term or trunc
        ep = None
        if done:
            ep = dict(self.shaper.stats, ret=self.ret, len=self.len, tags=self.tags, stage=int(self.stage_val.value))
            obs, feats = self.reset()
        else:
            feats = self.mem.update(obs, buttons)
        self.obs = obs
        return feats, r, done, ep, aux_targets(obs), label

    def close(self):
        for e in (self.env, self.fd_env):
            if e is not None:
                e.close()


def _worker(remote, wid, n, ctl, seed):
    import time

    time.sleep(0.4 * wid)  # stagger engine start-up across workers
    _hires_timer()
    import torch  # noqa: F401  (keeps thread settings consistent)

    torch.set_num_threads(1)
    slots = [_Slot(wid, wid * n + i, ctl, seed + 1000 * i) for i in range(n)]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                out = []
                for s in slots:
                    obs, feats = s.reset()
                    out.append((feats, aux_targets_safe(obs)))
                remote.send(out)
            elif cmd == "step":
                remote.send([s.step(c) for s, c in zip(slots, data)])
            elif cmd == "close":
                break
    finally:
        for s in slots:
            s.close()


def aux_targets_safe(obs):
    from rewards import aux_targets

    return aux_targets(obs)


class VecDoom:
    def __init__(self, workers: int, envs_per_worker: int, seed: int = 0, mode: int = 0):
        """mode 0: RL only; 1: RL + teacher labels (BC regulariser); 2: DAgger distillation."""
        ctx = mp.get_context("spawn")
        os.makedirs("_vizdoom", exist_ok=True)  # ViZDoom creates it on start; avoid a creation race
        self.stage = ctx.Value("i", 0)
        self.beta = ctx.Value("d", 0.0)
        self.mode = ctx.Value("i", mode)
        self.n = workers * envs_per_worker
        self.k = envs_per_worker
        self.remotes, self.procs = [], []
        for w in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, w, envs_per_worker, (self.stage, self.beta, self.mode), seed + 7919 * w), daemon=True)
            p.start()
            self.remotes.append(a)
            self.procs.append(p)

    @staticmethod
    def _stack(feats):
        return {k: np.stack([f[k] for f in feats]) for k in FEAT_KEYS}

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        res = [x for r in self.remotes for x in r.recv()]
        return self._stack([x[0] for x in res]), np.stack([x[1] for x in res])

    def step(self, choices: np.ndarray):
        for i, r in enumerate(self.remotes):
            r.send(("step", choices[i * self.k : (i + 1) * self.k]))
        res = [x for r in self.remotes for x in r.recv()]
        feats = self._stack([x[0] for x in res])
        rew = np.array([x[1] for x in res], np.float32)
        done = np.array([x[2] for x in res], np.float32)
        eps = [x[3] for x in res if x[3] is not None]
        aux = np.stack([x[4] for x in res])
        label = np.stack([x[5] for x in res])
        return feats, rew, done, eps, aux, label

    def set_stage(self, s: int):
        self.stage.value = int(s)

    def set_beta(self, b: float):
        self.beta.value = float(b)

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
