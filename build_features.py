"""Raw Big Data Bowl files -> one row per route run against man coverage.

Reads data/ (not in this repo) and writes outputs/routes_man.csv, the small
derived table that make_report.py needs. Runs in about a minute.

    python build_features.py
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

DATA = Path("data")
OUT = Path("outputs/routes_man.csv")

SNAP_EVENTS = {"ball_snap", "autoevent_ballsnap"}
THROW_EVENTS = {"pass_forward", "autoevent_passforward"}
KEYS = ["gameId", "playId"]
OPEN_YARDS = 2.0     # separation that counts as open
WINDOW_FRAMES = 3    # consecutive open frames (0.3 s) that count as a window
POS_GROUP = {"CB": "CB", "DB": "CB", "SS": "S", "FS": "S",
             "ILB": "LB", "OLB": "LB", "MLB": "LB", "LB": "LB"}


def load_snap_and_throw_frames(game_id: int) -> pd.DataFrame:
    """Return only the snap frame and the release frame of every play in a game,
    with coordinates flipped so every offense moves toward increasing x."""
    t = pd.read_csv(
        DATA / f"tracking/tracking_{game_id}.csv",
        usecols=["gameId", "playId", "nflId", "frameId", "team",
                 "playDirection", "x", "y", "event"],
    )
    left = t.playDirection == "left"
    t.loc[left, "x"] = 120 - t.loc[left, "x"]
    t.loc[left, "y"] = 53.3 - t.loc[left, "y"]
    # Snap and throw are each tagged two ways; take the first frame of either tag.
    snap = t[t.event.isin(SNAP_EVENTS)].groupby("playId").frameId.min().rename("snap")
    throw = t[t.event.isin(THROW_EVENTS)].groupby("playId").frameId.min().rename("throw")
    t = t.join(snap, on="playId").join(throw, on="playId")
    t = t[t.snap.notna() & t.throw.notna() & (t.throw > t.snap)]
    t["moment"] = np.select([t.frameId == t.snap, t.frameId == t.throw],
                            ["snap", "throw"], default="")
    return t.loc[t.moment != "", KEYS + ["nflId", "team", "x", "y", "moment", "snap", "throw"]]


def name_key(display_name: str) -> str:
    """'D.J. Moore' -> 'D.Moore', the form used in playDescription."""
    parts = re.sub(r" (Jr\.|Sr\.|II|III|IV|V)$", "", display_name).split(" ")
    return parts[0][0] + "." + " ".join(parts[1:])


def flag_targets(routes: pd.DataFrame) -> pd.Series:
    """There is no target column: the targeted receiver is the first route
    runner whose name appears after the word 'pass' in the play description."""
    def position_in_text(r):
        start = r.playDescription.find(" pass")
        return r.playDescription.find(r.key, start) if start >= 0 else -1

    where = routes.apply(position_in_text, axis=1)
    where = where.where(where >= 0)
    first = where.groupby([routes.gameId, routes.playId]).transform("min")
    return (where.notna() & (where == first)).astype(int)


def first_and_last_window(is_open: np.ndarray, seconds: np.ndarray) -> dict:
    """Summarise the runs of consecutive open frames between snap and release.
    A window is a run of at least WINDOW_FRAMES frames."""
    edges = np.diff(np.r_[0, is_open, 0])
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    windows = [(s, e) for s, e in zip(starts, ends) if e - s >= WINDOW_FRAMES]
    out = {"open_at_throw": int(is_open[-1]), "window": int(bool(windows)),
           "window_open_t": np.nan, "window_dur": np.nan, "window_close_t": np.nan, "window_closed": 0}
    if windows:
        s, e = windows[0]
        out.update(window_open_t=seconds[s], window_dur=(e - s) / 10)
        _, last_end = windows[-1]
        if last_end < len(is_open):          # shut again before the ball came out
            out.update(window_close_t=seconds[last_end], window_closed=1)
    return pd.Series(out)


def sticks_windows(routes: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Frame by frame from snap to release, on 3rd/4th-down WR/TE routes: is the
    receiver 2+ yards from the nearest coverage defender AND at or past the sticks?"""
    route_keys = KEYS + ["nflId"]
    per_frame = []
    for game_id, late in routes.groupby("gameId"):
        t = pd.read_csv(DATA / f"tracking/tracking_{game_id}.csv",
                        usecols=route_keys + ["frameId", "playDirection", "x", "y", "event"])
        t = t[t.playId.isin(late.playId)]
        t = t.join(t[t.event.isin(SNAP_EVENTS)].groupby("playId").frameId.min().rename("snap"), on="playId")
        t = t.join(t[t.event.isin(THROW_EVENTS)].groupby("playId").frameId.min().rename("throw"), on="playId")
        t = t[(t.frameId >= t.snap) & (t.frameId <= t.throw)].copy()
        left = t.playDirection == "left"
        t.loc[left, "x"] = 120 - t.loc[left, "x"]
        t.loc[left, "y"] = 53.3 - t.loc[left, "y"]
        rec = t.merge(late[route_keys + ["yardsToGo", "los_x"]], on=route_keys)
        dfn = t.merge(coverage, on=route_keys)[KEYS + ["frameId", "x", "y"]]
        pairs = rec.merge(dfn.rename(columns={"x": "dx", "y": "dy"}), on=KEYS + ["frameId"])
        pairs["dist"] = np.hypot(pairs.x - pairs.dx, pairs.y - pairs.dy)
        per_frame.append(pairs.groupby(route_keys + ["frameId", "snap", "x", "los_x", "yardsToGo"],
                                       as_index=False).dist.min())
    f = pd.concat(per_frame).sort_values(route_keys + ["frameId"])
    f["is_open"] = ((f.dist >= OPEN_YARDS) & (f.x - f.los_x >= f.yardsToGo)).astype(int)
    f["seconds"] = (f.frameId - f.snap) / 10
    return f.groupby(route_keys).apply(
        lambda g: first_and_last_window(g.is_open.values, g.seconds.values), include_groups=False).reset_index()


def main() -> None:
    games = pd.read_csv(DATA / "games.csv")
    plays = pd.read_csv(DATA / "plays.csv")
    players = pd.read_csv(DATA / "players.csv")
    pff = pd.read_csv(DATA / "pffScoutingData.csv",
                      usecols=KEYS + ["nflId", "pff_role", "pff_positionLinedUp"])

    frames = pd.concat(load_snap_and_throw_frames(g) for g in games.gameId)
    ball = frames[(frames.team == "football") & (frames.moment == "snap")]
    los = ball[KEYS + ["x"]].rename(columns={"x": "los_x"})
    frames = frames[frames.nflId.notna()].merge(pff, on=KEYS + ["nflId"])

    wide = frames.pivot_table(index=KEYS + ["nflId", "pff_role", "pff_positionLinedUp", "snap", "throw"],
                              columns="moment", values=["x", "y"]).reset_index()
    wide.columns = ["_".join(c).rstrip("_") for c in wide.columns]
    wide = wide.dropna(subset=["x_snap", "x_throw"])

    rec = wide[wide.pff_role == "Pass Route"]
    dfn = wide[wide.pff_role == "Coverage"][KEYS + ["nflId", "x_snap", "y_snap", "x_throw", "y_throw"]]
    dfn = dfn.rename(columns=lambda c: c if c in KEYS else "d_" + c)

    # Every receiver x coverage-defender pair on the play.
    pairs = rec[KEYS + ["nflId", "x_snap", "y_snap", "x_throw", "y_throw"]].merge(dfn, on=KEYS)
    pairs["dist_snap"] = np.hypot(pairs.x_snap - pairs.d_x_snap, pairs.y_snap - pairs.d_y_snap)
    pairs["dist_throw"] = np.hypot(pairs.x_throw - pairs.d_x_throw, pairs.y_throw - pairs.d_y_throw)
    by_route = pairs.groupby(KEYS + ["nflId"])

    # Separation = distance to the nearest coverage defender at release.
    nearest = pairs.loc[by_route.dist_throw.idxmin(), KEYS + ["nflId", "d_nflId", "dist_throw"]]
    nearest = nearest.rename(columns={"d_nflId": "release_def_id", "dist_throw": "sep"})
    # Matchup = the coverage defender nearest the receiver at the snap,
    # and how far that same defender is from him at release.
    lined_up = pairs.loc[by_route.dist_snap.idxmin(), KEYS + ["nflId", "d_nflId", "dist_snap", "dist_throw"]]
    lined_up = lined_up.rename(columns={"d_nflId": "snap_def_id", "dist_snap": "snap_def_dist",
                                        "dist_throw": "sep_snap_def"})

    def_pos = players.set_index("nflId").officialPosition.map(POS_GROUP).fillna("DL")
    routes = (rec.merge(nearest, on=KEYS + ["nflId"]).merge(lined_up, on=KEYS + ["nflId"])
                 .merge(los, on=KEYS)
                 .merge(players[["nflId", "displayName", "officialPosition"]], on="nflId")
                 .merge(plays, on=KEYS).merge(games[["gameId", "week"]], on="gameId"))
    routes["release_def_pos"] = routes.release_def_id.map(def_pos)
    routes["snap_def_pos"] = routes.snap_def_id.map(def_pos)
    routes["depth"] = routes.x_throw - routes.los_x
    routes["time_to_throw"] = (routes.throw - routes.snap) / 10
    routes["key"] = routes.displayName.map(name_key)
    routes["targeted"] = flag_targets(routes)

    # Thrown passes only, with exactly one identified target.
    routes = routes[routes.passResult.isin(["C", "I", "IN"])]
    n_plays = routes.groupby(KEYS).ngroups
    routes = routes[routes.groupby(KEYS).targeted.transform("sum") == 1]
    print(f"thrown plays with a release frame: {n_plays}; "
          f"target identified on {routes.groupby(KEYS).ngroups / n_plays:.1%}")

    routes["complete"] = ((routes.targeted == 1) & (routes.passResult == "C")).astype(int)
    routes["yards"] = np.where(routes.complete == 1, routes.prePenaltyPlayResult, 0)
    routes["past_sticks"] = (routes.depth >= routes.yardsToGo).astype(int)

    man = routes[(routes.pff_passCoverageType == "Man")
                 & routes.officialPosition.isin(["WR", "TE", "RB"])]
    # The Sticks Clock: when does a first-down window open, and does it last?
    late = man[man.down.isin([3, 4]) & man.officialPosition.isin(["WR", "TE"])]
    coverage = pff.loc[pff.pff_role == "Coverage", KEYS + ["nflId"]]
    man = man.merge(sticks_windows(late, coverage), on=KEYS + ["nflId"], how="left")

    cols = ["gameId", "playId", "week", "possessionTeam", "defensiveTeam", "down", "yardsToGo",
            "pff_passCoverage", "nflId", "displayName", "officialPosition", "pff_positionLinedUp",
            "depth", "past_sticks", "time_to_throw", "sep", "release_def_pos",
            "snap_def_pos", "snap_def_dist", "sep_snap_def", "targeted", "complete", "yards",
            "open_at_throw", "window", "window_open_t", "window_dur", "window_close_t", "window_closed"]
    OUT.parent.mkdir(exist_ok=True)
    man[cols].round(2).to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(man)} man-coverage routes on {man.groupby(KEYS).ngroups} plays; "
          f"{int(man.window.notna().sum())} 3rd/4th-down WR/TE routes with window fields")


if __name__ == "__main__":
    main()
