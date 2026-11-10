"""outputs/routes_man.csv -> the charts, tables and numbers quoted in the README.

Needs only the derived table that ships in this repo, not the raw tracking data.

    python make_report.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("outputs")
KEYS = ["gameId", "playId"]
OPEN_YARDS = 2.0      # separation from the nearest coverage defender at release
MIN_ROUTES = 50       # man-coverage routes needed to appear in the profile chart

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def load() -> pd.DataFrame:
    r = pd.read_csv(OUT / "routes_man.csv")
    r = r[r.officialPosition.isin(["WR", "TE"])].copy()
    r["open"] = (r.sep >= OPEN_YARDS).astype(int)
    r["open_past_sticks"] = r.open * r.past_sticks
    r["late_down"] = r.down.isin([3, 4])
    r["first_down"] = ((r.complete == 1) & (r.yards >= r.yardsToGo)).astype(int)
    return r


# ---------------------------------------------------------------- 1. proof
def proof_chart(routes: pd.DataFrame) -> dict:
    """Completion rate across continuous separation, and the case for 2 yards."""
    t = routes[routes.targeted == 1]
    edges = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8])
    b = t.groupby(pd.cut(t.sep, edges), observed=True).complete.agg(["mean", "size"])
    b["x"] = [iv.mid for iv in b.index]
    b["se"] = np.sqrt(b["mean"] * (1 - b["mean"]) / b["size"])
    avg = t.complete.mean()
    opened, covered = t[t.open == 1], t[t.open == 0]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.fill_between(b.x, b["mean"] - 1.64 * b.se, b["mean"] + 1.64 * b.se, color=BLUE, alpha=0.15, lw=0)
    ax.plot(b.x, b["mean"], color=BLUE, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5)
    ax.axhline(avg, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(7.9, avg - 0.035, f"average throw vs man: {pct(avg)}", ha="right", color=MUTED, fontsize=10)
    ax.axvline(OPEN_YARDS, color=INK, lw=1)
    ax.text(OPEN_YARDS + 0.12, 0.12, f"“open” = {OPEN_YARDS:g}+ yards\n{pct(opened.complete.mean())} complete "
            f"(n={len(opened)})", fontsize=10, va="bottom")
    ax.text(0.5, 0.93, f"covered (<{OPEN_YARDS:g} yd)\n{pct(covered.complete.mean())} complete\n"
            f"n={len(covered)}", fontsize=10, ha="left", va="top")
    ax.set(xlim=(0.4, 8), ylim=(0.1, 0.95), xlabel="Yards to the nearest coverage defender when the ball is released",
           ylabel="Completion rate")
    ax.yaxis.set_major_formatter(lambda v, _: pct(v))
    ax.set_title("Against man coverage, the first two yards of separation are worth the most")
    save(fig, "fig1_separation_vs_completion.png")

    sensitivity = {f"{thr:g}": round(t[t.sep >= thr].complete.mean() - t[t.sep < thr].complete.mean(), 3)
                   for thr in (1.5, 2.0, 2.5, 3.0)}
    return {"man_targets": len(t), "avg_completion": avg,
            "open_completion": opened.complete.mean(), "covered_completion": covered.complete.mean(),
            "open_yards_per_target": opened.yards.mean(), "covered_yards_per_target": covered.yards.mean(),
            "completion_by_half_yard": {str(k): round(v, 3) for k, v in b["mean"].items()},
            "completion_gap_by_threshold": sensitivity}


# ----------------------------------------------------------- 2. third down
def third_down(routes: pd.DataFrame) -> dict:
    """Open past the sticks on 3rd/4th down, and what a throw there is worth."""
    late = routes[routes.late_down]
    t = late[late.targeted == 1].copy()
    t["state"] = np.select([t.open_past_sticks == 1, t.open == 1],
                           ["Open, past the sticks", "Open, short of the sticks"], "Covered")
    order = ["Covered", "Open, short of the sticks", "Open, past the sticks"]
    s = t.groupby("state").first_down.agg(["mean", "size"]).loc[order]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.barh(order, s["mean"], height=0.55, color=BLUE)
    for bar, (_, row) in zip(bars, s.iterrows()):
        ax.text(row["mean"] + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{pct(row['mean'])}  ({int(row['size'])} targets)", va="center", fontsize=10)
    ax.set(xlim=(0, 0.85), xlabel="Share of targets that gained the first down")
    ax.xaxis.set_major_formatter(lambda v, _: pct(v))
    ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", colors=INK, length=0)
    ax.set_title("3rd/4th down vs man: where the target was when the ball came out")
    save(fig, "fig2_third_down.png")

    # Throws to a covered WR/TE while a WR/TE teammate was open past the sticks.
    def missed(df: pd.DataFrame) -> dict:
        per_play = df.groupby(KEYS).apply(lambda g: pd.Series({
            "target_open": g.loc[g.targeted == 1, "open"].max(),
            "mate_open": g.loc[g.targeted == 0, "open"].max(),
            "mate_open_past_sticks": g.loc[g.targeted == 0, "open_past_sticks"].max(),
        }), include_groups=False).dropna(subset=["target_open"]).fillna(0)
        cov = per_play[per_play.target_open == 0]
        return {"throws": len(per_play), "to_covered_target": len(cov),
                "and_teammate_open": int(cov.mate_open.sum()),
                "and_teammate_open_past_sticks": int(cov.mate_open_past_sticks.sum())}

    # Is the 3rd-down rate usable per player? Compare weeks 1-4 with 5-8.
    halves = late.assign(half=np.where(late.week <= 4, "a", "b")).groupby(["nflId", "half"]) \
                 .open_past_sticks.agg(["mean", "size"]).unstack()
    halves = halves[(halves["size"] >= 12).all(axis=1)]
    per_player = late.groupby("nflId").size()
    return {"routes": len(late), "plays": late.groupby(KEYS).ngroups,
            "open_past_sticks_rate": late.open_past_sticks.mean(),
            "open_past_sticks_rate_by_pos": late.groupby("officialPosition").open_past_sticks.mean().round(3).to_dict(),
            "first_down_rate_by_target_state": {k: round(v, 3) for k, v in s["mean"].items()},
            "targets_by_state": s["size"].to_dict(),
            "players_with_30_routes": int((per_player >= 30).sum()), "max_routes_one_player": int(per_player.max()),
            "split_half_r": round(halves["mean"].corr().iloc[0, 1], 2), "split_half_players": len(halves),
            "missed_all_downs": missed(routes), "missed_late_downs": missed(late)}


# -------------------------------------------------------------- 3. matchup
def matchup_grid(routes: pd.DataFrame) -> dict:
    """Open rate by who lined up over the receiver at the snap."""
    m = routes[routes.snap_def_pos.isin(["CB", "S", "LB"])]
    g = m.groupby(["officialPosition", "snap_def_pos"]).open.agg(["mean", "size"])
    rows, cols = ["WR", "TE"], ["CB", "S", "LB"]
    rate = g["mean"].unstack().loc[rows, cols]
    n = g["size"].unstack().loc[rows, cols]

    fig, ax = plt.subplots(figsize=(6.4, 3))
    ax.imshow(rate, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#cde2fb", "#104281"]),
              vmin=0.25, vmax=0.65, aspect="auto")
    for i in range(2):
        for j in range(3):
            dark = rate.iloc[i, j] > 0.45
            ax.text(j, i, f"{pct(rate.iloc[i, j])}\n{n.iloc[i, j]:,} routes", ha="center", va="center",
                    color="white" if dark else INK, fontsize=11)
    ax.set_xticks(range(3), ["Cornerback", "Safety", "Linebacker"]); ax.set_yticks(range(2), rows)
    ax.set_xticks(np.arange(-0.5, 3), minor=True); ax.set_yticks(np.arange(-0.5, 2), minor=True)
    ax.grid(visible=False); ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0, colors=INK)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlabel("Nearest coverage defender at the snap")
    ax.set_title("Share of routes open at release, by who lined up over him")
    save(fig, "fig3_matchup_grid.png")
    same = (routes.snap_def_pos == routes.release_def_pos).mean()
    return {"open_rate": rate.round(3).to_dict("index"), "routes": n.to_dict("index"),
            "snap_and_release_defender_same_position_group": round(same, 3)}


# ------------------------------------------------------------- 4. profiles
def profiles(routes: pd.DataFrame) -> dict:
    """Per-receiver open rate, shrunk toward the position average."""
    p = routes.groupby(["nflId", "displayName", "officialPosition", "possessionTeam"]).agg(
        routes=("open", "size"), open_n=("open", "sum"), open_rate=("open", "mean"),
        target_rate=("targeted", "mean"), avg_depth=("depth", "mean"),
        late_routes=("late_down", "sum"),
        late_open_past_sticks=("open_past_sticks", lambda s: s[routes.loc[s.index, "late_down"]].sum()),
    ).reset_index().rename(columns={"displayName": "player", "officialPosition": "pos", "possessionTeam": "team"})
    q = p[p.routes >= MIN_ROUTES].copy()

    # Empirical-Bayes shrinkage: how much of the spread between players is real?
    pos_mean = routes.groupby("officialPosition").open.mean()
    noise = (q.open_rate * (1 - q.open_rate) / q.routes).mean()
    real_var = max(q.open_rate.var() - noise, 1e-4)
    m = routes.open.mean()
    prior_n = m * (1 - m) / real_var - 1
    q["open_rate_adj"] = (q.open_n + prior_n * q.pos.map(pos_mean)) / (q.routes + prior_n)
    q["se"] = np.sqrt(q.open_rate_adj * (1 - q.open_rate_adj) / (q.routes + prior_n))
    q = q.sort_values("open_rate_adj", ascending=False)
    q.drop(columns=["nflId", "open_n"]).round(3).to_csv(OUT / "receiver_profiles.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 0.24 * len(q) + 1.4))
    y = np.arange(len(q))
    for pos, color in (("WR", BLUE), ("TE", ORANGE)):
        k = (q.pos == pos).values
        ax.hlines(y[k], (q.open_rate_adj - 1.64 * q.se)[k], (q.open_rate_adj + 1.64 * q.se)[k], color=color, lw=2, alpha=0.3)
        ax.plot(q.open_rate_adj[k], y[k], "o", ms=7, color=color, mec=SURFACE, mew=1.5, label=pos)
    ax.set_yticks(y, [f"{n}  ({t}, {r})" for n, t, r in zip(q.player, q.team, q.routes)], fontsize=9, color=INK)
    ax.set_ylim(len(q) - 0.4, -0.6); ax.tick_params(axis="y", length=0); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(lambda v, _: pct(v))
    ax.set_xlabel(f"Share of man-coverage routes with {OPEN_YARDS:g}+ yards of separation at release\n"
                  "(adjusted toward position average; bar = 90% interval; team, routes in brackets)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("How often each receiver is already open vs man when the ball comes out", fontsize=12)
    save(fig, "fig4_receiver_profiles.png")

    def split_half(min_per_half: int):
        h = routes.assign(half=np.where(routes.week <= 4, "a", "b")).groupby(["nflId", "half"]).open \
                  .agg(["mean", "size"]).unstack()
        h = h[(h["size"] >= min_per_half).all(axis=1)]
        return {"players": len(h), "r": round(h["mean"].corr().iloc[0, 1], 2)}

    return {"players": len(q), "raw_range": [round(q.open_rate.min(), 3), round(q.open_rate.max(), 3)],
            "prior_routes": round(prior_n, 1), "position_mean": pos_mean.round(3).to_dict(),
            "weeks_1_4_vs_5_8": {"min_25_per_half": split_half(25), "min_20_per_half": split_half(20)},
            "top": q.head(8)[["player", "pos", "open_rate", "open_rate_adj"]].round(3).to_dict("records"),
            "bottom": q.tail(8)[["player", "pos", "open_rate", "open_rate_adj"]].round(3).to_dict("records")}


def main() -> None:
    routes = load()
    findings = {"open_definition_yards": OPEN_YARDS, "wr_te_man_routes": len(routes),
                "proof": proof_chart(routes), "third_down": third_down(routes),
                "matchup": matchup_grid(routes), "profiles": profiles(routes)}
    (OUT / "findings.json").write_text(json.dumps(findings, indent=2, default=float))
    print(json.dumps(findings, indent=2, default=float))


if __name__ == "__main__":
    main()
