"""Play DOOM yourself (human spectator mode) on a generated level or a Freedoom 2 map.

    python training/play.py                      # generated keycard level (stage 2)
    python training/play.py --stage 4 --seed 9   # harder generated level
    python training/play.py --freedoom MAP01     # real Freedoom 2 campaign map
    python training/play.py --wad maps/two_rooms.wad

Controls: W/A/S/D move, mouse turn, left click fire, E or Space use (doors), 1-7 weapons,
Shift run, Esc menu / quit.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

import vizdoom as vd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

BINDS = ("+bind w +forward +bind s +back +bind a +moveleft +bind d +moveright "
         "+bind e +use +bind space +use +bind mouse1 +attack +bind shift +speed "
         "+bind leftarrow +left +bind rightarrow +right +bind uparrow +forward +bind downarrow +back "
         "+freelook 1 +m_pitch 0 +sensitivity 2.5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--wad", default=None)
    ap.add_argument("--map", default="MAP01")
    ap.add_argument("--freedoom", default=None, help="play a Freedoom 2 map, e.g. MAP01")
    ap.add_argument("--skill", type=int, default=3)
    a = ap.parse_args()
    if os.name == "nt":
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)

    iwad = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
    g = vd.DoomGame()
    g.set_doom_game_path(iwad)
    if a.freedoom:
        g.set_doom_map(a.freedoom)
    else:
        wad = a.wad
        if wad is None:
            from mapgen import ascii_map, spec_for_stage, write_level

            spec = spec_for_stage(a.stage, random.Random(a.seed))
            wad = str(Path(tempfile.gettempdir()) / f"play_{a.stage}_{a.seed}.wad")
            L = write_level(wad, spec, a.seed)
            print(f"level: {spec.tags}  (P start, y/b/r keys, Y/B/R locked doors, D door, E exit, m monster)")
            print(ascii_map(L))
        g.set_doom_scenario_path(str(Path(wad).resolve()))
        g.set_doom_map(a.map)
    g.set_doom_skill(a.skill)
    g.set_mode(vd.Mode.ASYNC_SPECTATOR)
    g.set_screen_resolution(vd.ScreenResolution.RES_1024X768)
    g.set_window_visible(True)
    g.set_sound_enabled(True)
    g.set_render_hud(True)
    g.set_render_crosshair(True)
    g.set_render_messages(True)
    g.set_console_enabled(True)
    g.set_episode_timeout(0)
    g.add_game_args(BINDS)
    g.init()
    g.new_episode()
    while not g.is_episode_finished():
        g.advance_action()
    print("exit reached!" if not g.is_player_dead() else "you died")
    print(f"kills {g.get_game_variable(vd.GameVariable.KILLCOUNT):.0f}, "
          f"time {g.get_episode_time() / 35:.0f}s")
    g.close()


if __name__ == "__main__":
    main()
