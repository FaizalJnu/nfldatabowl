# Open When It Has to Come Out

**On 3rd down against man coverage, who is likely to be open when the ball has to come out?**

We measured every WR and TE route run against man coverage in Weeks 1–8 of 2021 (7,823 routes, 122 games) and asked one thing at the frame the ball leaves the quarterback's hand: does the receiver have 2+ yards on the nearest coverage defender? A target who does is completed 69% of the time for 9.1 yards; a target who doesn't, 41% for 6.9. On 3rd and 4th down a target who is open *and* past the sticks converts 66% of the time against 39% for a covered one, yet 486 of the 850 third- and fourth-down throws went to a covered receiver, and on 166 of those a teammate was open past the sticks. An offensive coordinator or advance scout can use the matchup grid and receiver profiles below to decide who to isolate on the money down: wide receivers are open 34% of the time when a cornerback lines up over them, 49% against a safety and 61% against a linebacker, while tight ends get nothing from a safety (32%). Separation is measured only at release because the tracking ends there, defenders are inferred from position rather than assignment, and eight weeks is too few routes to rank individual players, so read the profiles as how a receiver gets open, not who is best.

## What a coach sees

**1. Two yards at release is the line between a bad throw and a good one.**
Completion rate climbs steeply up to about 2 yards of separation and slowly after it. The curve crosses the average throw against man (54%) just before 2 yards, so we call 2+ yards "open": every half-yard band above it beats the average and every band below 1.5 falls short. The gap between open and covered is 28–31 points whether the line is drawn at 1.5, 2, 2.5 or 3 yards, so the finding does not depend on the exact cut.

![Completion rate by separation at release](outputs/fig1_separation_vs_completion.png)

**2. On the money down, open is not enough: it has to be open past the sticks.**
A WR or TE is open past the first-down line on 16% of 3rd/4th-down man routes (WRs 18%, TEs 12%). Throwing to one converts two times in three. Open but short of the sticks is barely better than covered.

![First-down rate by where the target was at release](outputs/fig2_third_down.png)

**3. The matchup to hunt is a wide receiver on a safety or linebacker.**
The defender is whoever in coverage lined up nearest the receiver at the snap; "open" is still 2+ yards from the nearest defender at release.

![Open rate by receiver position and defender lined up over him](outputs/fig3_matchup_grid.png)

**4. Receiver profiles: who is already open when the ball comes out.**
Receivers with 50+ man-coverage routes, all downs. The top of the chart is early separators (Waller, Kupp, Beasley, Hockenson). The bottom is mostly receivers who win later or at the catch point (Chase, Mike Williams, Evans, Higgins), which this data cannot see. It is a description of style, not a ranking of quality: the rate has almost no relationship with yards per route.

![Receiver profiles](outputs/fig4_receiver_profiles.png)

## Run it

```
pip install -r requirements.txt
python make_report.py        # rebuilds every figure and number from outputs/routes_man.csv
python build_features.py     # optional: rebuilds routes_man.csv from the raw data in data/
```

The raw tracking data is not in this repo. `outputs/routes_man.csv` is the derived per-route table (one row per route against man coverage), so `make_report.py` runs without it. Every number quoted above is in `outputs/findings.json`, and the profile table is `outputs/receiver_profiles.csv`.

## Method

- **Release frame:** the first `pass_forward` or `autoevent_passforward` event. Coordinates are flipped by `playDirection`.
- **Open:** 2+ yards from the nearest player PFF tagged as `Coverage`, at the release frame.
- **Past the sticks:** the receiver's depth beyond the ball's position at the snap is at least `yardsToGo`.
- **Plays:** PFF coverage type `Man`, pass thrown (complete, incomplete or intercepted), target found in the play description.
- **Target:** parsed from `playDescription`, because there is no target column. It was found on 94.7% of thrown plays.
- **Profiles:** each receiver's open rate is pulled toward his position's average in proportion to how few routes he has (empirical Bayes), with a 90% interval.

## Assumptions and limitations

- Tracking stops at the throw. Receivers who separate while the ball is in the air, or win contested catches, look worse here than they are.
- "The defender lined up over him" is the nearest coverage defender at the snap, not the assigned man. It is the same position group as the nearest defender at release on 80% of routes.
- Individual numbers are noisy. Comparing Weeks 1–4 with 5–8, the open rate correlates at 0.60 for the 24 receivers with 25+ routes in each half, but only 0.21 for the 45 with 20+.
- The past-the-sticks rate is reported for the league only. No receiver has more than 41 third- or fourth-down man routes, and the per-player rate does not hold up between the two halves (r = 0.17).
- "A teammate was open" does not mean the quarterback could have thrown it: we do not model pressure, the read progression or throwing lanes.
- Sacks, scramble plays and throws with no tagged release are excluded. The rate is not adjusted for route depth or the quality of the defender.
