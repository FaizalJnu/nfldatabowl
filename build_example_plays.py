"""Raw tracking for two example plays -> outputs/example_plays.json.

Reads data/ (not in this repo) for only the two plays below, from snap to throw,
and saves the paths make_report.py needs to draw "The window on a real play".

    python build_example_plays.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from build_features import DATA, KEYS, OPEN_YARDS, SNAP_EVENTS, THROW_EVENTS, WINDOW_FRAMES

OUT = Path("outputs/example_plays.json")
PLAYS = [
    {"gameId": 2021100308, "playId": 4190,
     "story": "Thrown into the window: the target is open past the sticks when the ball comes out"},
    {"gameId": 2021091206, "playId": 2011,
     "story": "Thrown after the window closed: the target is covered again and a teammate is open"},
]


def in_window(is_open: np.ndarray) -> list:
    """Flag the frames that belong to a run of WINDOW_FRAMES or more open frames."""
    flags = np.zeros(len(is_open), dtype=int)
    edges = np.diff(np.r_[0, is_open, 0])
    for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        if e - s >= WINDOW_FRAMES:
            flags[s:e] = 1
    return flags.tolist()


def extract(game_id: int, play_id: int, story: str, plays, players, pff, routes) -> dict:
    play = plays[(plays.gameId == game_id) & (plays.playId == play_id)].iloc[0]
    t = pd.read_csv(DATA / f"tracking/tracking_{game_id}.csv")
    t = t[t.playId == play_id]
    snap = t[t.event.isin(SNAP_EVENTS)].frameId.min()
    throw = t[t.event.isin(THROW_EVENTS)].frameId.min()
    t = t[(t.frameId >= snap) & (t.frameId <= throw)].sort_values("frameId").copy()
    if (t.playDirection == "left").all():          # make every offense move toward increasing x
        t["x"], t["y"] = 120 - t.x, 53.3 - t.y

    ball = t[t.team == "football"]
    los_x = float(ball.x.iloc[0])
    sticks_x = los_x + float(play.yardsToGo)
    roles = pff[(pff.gameId == game_id) & (pff.playId == play_id)].set_index("nflId")
    names = players.set_index("nflId")
    target = routes[(routes.gameId == game_id) & (routes.playId == play_id) & (routes.targeted == 1)].iloc[0]

    people = t[t.nflId.notna()]
    coverage = people[people.nflId.map(roles.pff_role) == "Coverage"]
    out_players = []
    for nfl_id, p in people.groupby("nflId"):
        role, pos = roles.pff_role[nfl_id], names.officialPosition[nfl_id]
        rec = {"name": names.displayName[nfl_id], "position": pos, "role": role,
               "side": "offense" if p.team.iloc[0] == play.possessionTeam else "defense",
               "x": p.x.round(1).tolist(), "y": p.y.round(1).tolist(),
               "target": bool(nfl_id == target.nflId)}
        if role == "Pass Route" and pos in ("WR", "TE"):
            # Same test as the Sticks Clock: 2+ yards from every coverage defender, at or past the sticks.
            m = p[["frameId", "x", "y"]].merge(coverage[["frameId", "x", "y"]], on="frameId", suffixes=("", "_d"))
            sep = np.hypot(m.x - m.x_d, m.y - m.y_d).groupby(m.frameId).min()
            is_open = ((sep.values >= OPEN_YARDS) & (p.x.values >= sticks_x)).astype(int)
            rec["in_window"] = in_window(is_open)
        out_players.append(rec)

    return {"gameId": game_id, "playId": play_id, "story": story,
            "offense": play.possessionTeam, "defense": play.defensiveTeam,
            "down": int(play.down), "yardsToGo": int(play.yardsToGo),
            "time_to_throw": round((throw - snap) / 10, 1), "passResult": play.passResult,
            "yards": int(play.prePenaltyPlayResult), "touchdown": "TOUCHDOWN" in play.playDescription,
            "description": play.playDescription, "los_x": round(los_x, 1), "sticks_x": round(sticks_x, 1),
            "ball_at_throw": [round(float(ball.x.iloc[-1]), 1), round(float(ball.y.iloc[-1]), 1)],
            "players": out_players}


def main() -> None:
    plays = pd.read_csv(DATA / "plays.csv")
    players = pd.read_csv(DATA / "players.csv")
    pff = pd.read_csv(DATA / "pffScoutingData.csv", usecols=KEYS + ["nflId", "pff_role"])
    routes = pd.read_csv("outputs/routes_man.csv")
    examples = [extract(p["gameId"], p["playId"], p["story"], plays, players, pff, routes) for p in PLAYS]
    OUT.write_text(json.dumps(examples))
    for e in examples:
        print(e["offense"], "vs", e["defense"], e["description"])
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
