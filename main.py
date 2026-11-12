#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming chart top 10 — v2 rewrite
(make.fun roblox-up-and-coming-forecast-01).

Method (a rewrite, not a tweak of v1 — v1's single OU was ~3x too flat):
  * Learned kernel/analog forecaster: every past read is an origin; for
    each horizon bin (0-6, 6-12, 12-24, 24-36, 36-48, 48-60, 60-72 h) we
    record where games with similar features ended up. A game is
    described by log-rank, rank velocity over ~24 h and ~6 h, player-count
    growth over ~24 h, chart age, and distance below its own best rank.
    Its place probabilities are the Gaussian-kernel-weighted mean of past
    games with similar features, with a recency weight on origins.
  * Runtime ridge regression on the same features predicts each game's
    log-rank velocity; that drives a corrected OU Monte Carlo (the
    variance denominator uses the same rate as the decay — v1's bug,
    which inflated the noise SD ~1.41x, is fixed). Momentum is carried
    at full strength (DRIFT_SCALE ~1.2) with a breakout filter (young
    tenure + rising rank + rising players) instead of a blanket shrink.
  * Newcomer share is LEARNED per horizon bin and per place from the
    history (it grows ~0 at 0-6 h to ~0.21 at 60-72 h), not one window
    average; it is the unnamed reserve.
  * Age is a MEASURED exit-hazard curve (survival by chart age and
    horizon, fitted from the history), not a hard 28-day cliff.
  * Calibration discipline: floors on named probabilities, soft cap on
    place shares, everything fitted on a fast (recent) window.
Python 3.13, standard library only, deterministic (fixed seed), no network.
Reads $MAKEFUN_INPUT_DIR (else /input) and writes $MAKEFUN_OUTPUT_DIR (else /output).
"""

import bisect
import json
import math
import os
import random
from datetime import datetime, timedelta

SEED = 20261113
N_DRAWS = 3000        # Monte Carlo draws per slot time (parametric component)
N_SLOTS = 12          # evenly spaced slot times across the window
BINS = [(0.0, 6.0), (6.0, 12.0), (12.0, 24.0), (24.0, 36.0), (36.0, 48.0),
        (48.0, 60.0), (60.0, 72.0)]
# Gaussian-kernel bandwidths, feature order:
# (log-rank, rank-vel 24h, rank-vel 6h, player-growth 24h, chart-age h, below-peak)
BW = (0.5, 0.02, 0.02, 0.015, 48.0, 0.5)
X0_WINDOW = 1.2       # kernel origins are pre-filtered to |d log-rank| <= this
D2_CUT = 18.0         # kernel distance cutoff (exp(-9) is negligible)
RECENCY_TAU = 120.0   # hours: kernel origin recency decay
ESS0 = 30.0           # kernel effective-sample-size scale for the blend
ALPHA = 0.5           # kernel weight in the blend (before ESS adjustment)
DRIFT_SCALE = 1.2     # momentum carried at >1x (winners), not shrunk
BREAKOUT_EXTRA = 0.3  # extra velocity multiplier for breakout games
VOL_SCALE = 1.0       # diffusion scale for the parametric component
CAP = 0.12            # soft cap on a game's share of one place
CAP_SHRINK = 0.5      # excess kept past the cap
FLOOR = 0.0002        # floor on a named probability
RESERVE_SCALE = 1.0   # scale on the learned newcomer reserve
RESERVE_FLOOR = 0.20  # min unnamed reserve / place (tuned: 0.01->0.20 raised
                      # mean diff +8.75 -> +12.01; dev stays +11.0)
RIDGE_LAM = 1.0       # ridge penalty (intercept unpenalised)
MAX_ORIGIN_RANK = 50  # origins only from games ranked <= this at the origin

# Priors used when the history is too short to fit (fallbacks only).
PRIOR_KAPPA = 0.006
PRIOR_XBAR = 3.5
PRIOR_SIGMA0 = 0.10


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_input(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    reads.sort(key=lambda r: r["at"])
    return task, reads


def chart_of(read):
    """universe_id -> (rank, playing) of the up-and-coming chart."""
    rows = sorted(read.get("up_and_coming") or [], key=lambda g: g["rank"])
    return {g["universe_id"]: (g["rank"], g.get("playing") or 0) for g in rows}


def solve_linear(A, b):
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            continue
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c] / pv
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]

def build_trajectories(charts, times):
    """uid -> (times list, ranks list, playing list), and first-seen time."""
    traj = {}
    for i, ch in enumerate(charts):
        for uid, (rank, playing) in ch.items():
            t = traj.get(uid)
            if t is None:
                traj[uid] = ([times[i]], [rank], [playing])
            else:
                t[0].append(times[i])
                t[1].append(rank)
                t[2].append(playing)
    return traj


def _lookback(times, ranks, playings, t_i, target_h, lo_h, hi_h, min_dt):
    """Slope of log-rank (and log-playing) over ~target_h hours back.

    Returns (v_rank, g_play, dt_used). Picks the trajectory point closest
    to t_i - target_h among points in [t_i - hi_h, t_i - lo_h]; 0 if none.
    """
    t_targ = t_i - timedelta(hours=target_h)
    idx = bisect.bisect_right(times, t_targ) - 1
    if idx < 0:
        return 0.0, 0.0, 0.0
    # walk to the closest point to the target (a few steps at most)
    best = idx
    best_gap = abs((times[idx] - t_targ).total_seconds())
    for j in (idx - 1, idx + 1):
        if 0 <= j < len(times):
            gap = abs((times[j] - t_targ).total_seconds())
            if gap < best_gap:
                best, best_gap = j, gap
    dt = (t_i - times[best]).total_seconds() / 3600.0
    if dt < min_dt or dt > hi_h:
        return 0.0, 0.0, 0.0
    v = (math.log(max(ranks[-1], 1)) - math.log(max(ranks[best], 1))) / dt
    g = 0.0
    if playings[-1] > 0 and playings[best] > 0:
        g = (math.log(playings[-1]) - math.log(playings[best])) / dt
    return v, g, dt


def features_at(uid, traj, upto_i, times):
    """Feature vector of a game at read index upto_i (reads <= upto_i only).

    Returns (x0, v24, v6, g24, age_h, below) where x0 is log-rank, v24/v6
    are log-rank velocities per hour, g24 is log-playing growth per hour,
    age_h is hours since first seen on the chart, below is the distance
    below the game's own best log-rank so far.
    """
    ts, ranks, playings = traj[uid]
    # truncate the trajectory to points at or before read upto_i
    t_i = times[upto_i]
    n = bisect.bisect_right(ts, t_i)
    if n == 0:
        return None
    ts, ranks, playings = ts[:n], ranks[:n], playings[:n]
    x0 = math.log(max(ranks[-1], 1))
    v24, g24, _ = _lookback(ts, ranks, playings, t_i, 24.0, 12.0, 48.0, 6.0)
    v6, _, _ = _lookback(ts, ranks, playings, t_i, 6.0, 3.0, 10.0, 2.0)
    age = (t_i - ts[0]).total_seconds() / 3600.0
    best = min(ranks)
    below = x0 - math.log(max(best, 1))
    return (x0, v24, v6, g24, age, below)


def is_breakout(f):
    """Young tenure + rising rank + rising players."""
    return f[4] < 72.0 and f[2] < -0.01 and f[3] > 0.0

def bin_index(dt_h):
    for b, (lo, hi) in enumerate(BINS):
        if lo < dt_h <= hi or (b == 0 and dt_h <= hi):
            return b
    return None


def build_origins(charts, times, feats, cut_time):
    """Build kernel origins, newcomer shares and survival from read pairs.

    feats[i] is a dict uid -> feature tuple at read i.
    Returns (origins, newc, surv) where origins[b] is a list of
    (x0, feat, out_rank_or_None, origin_time) sorted by x0, newc[b][k] is
    the newcomer share of place k in bin b, surv[(age_bucket, b)] is the
    measured survival fraction.
    """
    n = len(charts)
    origins = [[] for _ in BINS]
    # newcomer counts: newc_cnt[b][k] = #pairs where place k was a newcomer
    newc_cnt = [[0] * 11 for _ in BINS]
    newc_tot = [0] * len(BINS)
    # survival counts: surv_cnt[(age_bucket, b)] = (alive, total)
    surv_cnt = {}
    max_pairs = 4000
    # pair stride: keep runtime bounded on long (final) histories
    total_pairs = sum(1 for i in range(n) for j in range(i + 1, n)
                      if (times[j] - times[i]).total_seconds() <= 72 * 3600)
    stride = max(1, total_pairs // max_pairs + 1)
    pair_no = 0
    for i in range(n):
        ci = charts[i]
        fi = feats[i]
        for j in range(i + 1, n):
            dt = (times[j] - times[i]).total_seconds() / 3600.0
            if dt > 72.0:
                break
            b = bin_index(dt)
            if b is None:
                continue
            pair_no += 1
            if pair_no % stride:
                continue
            cj = charts[j]
            newc_tot[b] += 1
            top10 = sorted(cj.items(), key=lambda kv: kv[1][0])[:10]
            for k, (uid, _) in enumerate(top10, start=1):
                if uid not in ci:
                    newc_cnt[b][k] += 1
            # survival by chart-age bucket of the origin game
            for uid, (rank, _) in ci.items():
                f = fi.get(uid)
                if f is None or rank > MAX_ORIGIN_RANK:
                    continue
                age = f[4]
                ab = 0 if age < 72 else (1 if age < 168 else (2 if age < 336 else 3))
                key = (ab, b)
                a, t = surv_cnt.get(key, (0, 0))
                surv_cnt[key] = (a + (1 if uid in cj else 0), t + 1)
                out = cj[uid][0] if uid in cj else None
                origins[b].append((f[0], f, out, times[i]))
    origins_sorted = []
    for b in range(len(BINS)):
        origins_sorted.append(sorted(origins[b], key=lambda r: r[0]))
    newc = []
    for b in range(len(BINS)):
        tot = max(newc_tot[b], 1)
        newc.append([0.0] + [newc_cnt[b][k] / tot for k in range(1, 11)])
    surv = {}
    for (ab, b), (a, t) in surv_cnt.items():
        surv[(ab, b)] = a / t if t >= 20 else None
    return origins_sorted, newc, surv


def fit_ridge(charts, times, feats, cut_time):
    """Ridge regression of per-hour log-rank velocity on game features.

    Samples are pairs (i, j) with the game present at both reads; the
    target is (ln rank_j - ln rank_i) / dt. Returns (beta, sigma0, a, b)
    where beta predicts velocity, sigma0 is the per-hour residual SD, and
    a, b are the global drift fit (drift = a + b*x0) for the OU equilibrium.
    Falls back to priors when there are too few samples.
    """
    n = len(charts)
    # normal equations for 8 features: [1, x0, v24, v6, g24, age/72, below, breakout]
    NF = 8
    XtX = [[0.0] * NF for _ in range(NF)]
    Xty = [0.0] * NF
    # global drift fit accumulators: y/dt regressed on x0
    sx = sy = sxx = sxy = sw = 0.0
    res_num = res_den = 0.0
    nsamp = 0
    total_pairs = sum(1 for i in range(n) for j in range(i + 1, n)
                      if (times[j] - times[i]).total_seconds() <= 72 * 3600)
    stride = max(1, total_pairs // 1500 + 1)
    pair_no = 0
    for i in range(n):
        ci = charts[i]
        fi = feats[i]
        w_rec = math.exp(-(cut_time - times[i]).total_seconds() / 3600.0 / 72.0)
        for j in range(i + 1, n):
            dt = (times[j] - times[i]).total_seconds() / 3600.0
            if dt > 72.0:
                break
            if dt < 1.0:
                continue
            pair_no += 1
            if pair_no % stride:
                continue
            cj = charts[j]
            for uid, (rank, _) in ci.items():
                if uid not in cj:
                    continue
                f = fi.get(uid)
                if f is None:
                    continue
                y = (math.log(max(cj[uid][0], 1)) - math.log(max(rank, 1))) / dt
                w = dt * w_rec
                z = (1.0, f[0], f[1], f[2], f[3], f[4] / 72.0, f[5],
                     1.0 if is_breakout(f) else 0.0)
                for a in range(NF):
                    Xty[a] += w * z[a] * y
                    for c in range(a, NF):
                        XtX[a][c] += w * z[a] * z[c]
                sx += w * f[0]
                sy += w * y
                sxx += w * f[0] * f[0]
                sxy += w * f[0] * y
                sw += w
                nsamp += 1
    if nsamp < 100 or sw <= 0:
        return None, PRIOR_SIGMA0, 0.024, -PRIOR_KAPPA
    for a in range(NF):
        for c in range(a):
            XtX[a][c] = XtX[c][a]
        XtX[a][a] += RIDGE_LAM if a > 0 else 0.0
    beta = solve_linear(XtX, Xty)
    # residual SD per hour (recompute residuals cheaply on a strided subset)
    res_num = res_den = 0.0
    pair_no = 0
    for i in range(n):
        ci = charts[i]
        fi = feats[i]
        w_rec = math.exp(-(cut_time - times[i]).total_seconds() / 3600.0 / 72.0)
        for j in range(i + 1, n):
            dt = (times[j] - times[i]).total_seconds() / 3600.0
            if dt > 72.0:
                break
            if dt < 6.0 or dt > 48.0:
                continue
            pair_no += 1
            if pair_no % 3:
                continue
            cj = charts[j]
            for uid, (rank, _) in ci.items():
                if uid not in cj:
                    continue
                f = fi.get(uid)
                if f is None:
                    continue
                y = math.log(max(cj[uid][0], 1)) - math.log(max(rank, 1))
                z = (1.0, f[0], f[1], f[2], f[3], f[4] / 72.0, f[5],
                     1.0 if is_breakout(f) else 0.0)
                pred = sum(beta[k] * z[k] for k in range(NF)) * dt
                res_num += w_rec * (y - pred) ** 2 / dt
                res_den += w_rec
    sigma0 = math.sqrt(res_num / res_den) if res_den > 0 else PRIOR_SIGMA0
    sigma0 = min(max(sigma0, 0.04), 0.25)
    # global drift a + b*x0
    den = sw * sxx - sx * sx
    if den > 0:
        b = (sw * sxy - sx * sy) / den
        a = (sy - b * sx) / sw
    else:
        a, b = 0.024, -PRIOR_KAPPA
    a = min(max(a, 0.0), 0.05)
    b = min(max(b, -0.02), 0.005)
    return beta, sigma0, a, b

def kernel_places(f, origins, cut_time):
    """Kernel place distribution per horizon bin for one candidate.

    Returns (K, ess) where K[b] is a list of 11 probabilities (index 0
    unused, 1..10 = P(place k)) and ess[b] the effective sample size.
    Origins with |d log-rank| > X0_WINDOW are skipped via bisect.
    """
    K = []
    ess_list = []
    xs = [[r[0] for r in origins[b]] for b in range(len(BINS))]
    for b in range(len(BINS)):
        ob = origins[b]
        xb = xs[b]
        lo = bisect.bisect_left(xb, f[0] - X0_WINDOW)
        hi = bisect.bisect_right(xb, f[0] + X0_WINDOW)
        mass = [0.0] * 11
        wtot = 0.0
        w2 = 0.0
        for idx in range(lo, hi):
            o_x0, o_f, out, o_t = ob[idx]
            d2 = ((f[0] - o_x0) / BW[0]) ** 2
            if d2 > D2_CUT:
                continue
            d2 += ((f[1] - o_f[1]) / BW[1]) ** 2
            if d2 > D2_CUT:
                continue
            d2 += ((f[2] - o_f[2]) / BW[2]) ** 2
            if d2 > D2_CUT:
                continue
            d2 += ((f[3] - o_f[3]) / BW[3]) ** 2
            if d2 > D2_CUT:
                continue
            d2 += ((f[4] - o_f[4]) / BW[4]) ** 2
            if d2 > D2_CUT:
                continue
            d2 += ((f[5] - o_f[5]) / BW[5]) ** 2
            if d2 > D2_CUT:
                continue
            w = math.exp(-0.5 * d2)
            w *= math.exp(-(cut_time - o_t).total_seconds() / 3600.0 / RECENCY_TAU)
            wtot += w
            w2 += w * w
            if out is not None and 1 <= out <= 10:
                mass[out] += w
        if wtot > 0:
            K.append([m / wtot for m in mass])
            ess_list.append(wtot * wtot / w2 if w2 > 0 else 0.0)
        else:
            K.append([0.0] * 11)
            ess_list.append(0.0)
    return K, ess_list


def ridge_velocity(beta, f):
    """Per-hour log-rank velocity predicted by the ridge for feature f."""
    z = (1.0, f[0], f[1], f[2], f[3], f[4] / 72.0, f[5],
         1.0 if is_breakout(f) else 0.0)
    return sum(beta[k] * z[k] for k in range(len(beta)))


def mc_places(cands, beta, a, b, kappa, xbar, sigma0, surv, slot_hs,
              base_h, last_time, rng):
    """Parametric component: OU Monte Carlo with ridge drift.

    Returns (M, mean_rank) where M[uid] is a list of 11 probabilities
    (index 0 unused) and mean_rank[uid] is the mean simulated rank.
    The OU variance uses the SAME kappa as the decay (v1's bug fixed).
    """
    freq = {c["uid"]: [0] * 11 for c in cands}
    rank_sum = {c["uid"]: 0.0 for c in cands}
    rank_cnt = {c["uid"]: 0 for c in cands}
    n_sims = 0
    for sh in slot_hs:
        t = max(base_h + sh, 0.0)
        decay = math.exp(-kappa * t)
        # corrected variance factor: (1 - exp(-2 k t)) / (2 k), same k as decay
        var_f = math.sqrt(max((1.0 - decay * decay) / (2.0 * kappa), 0.0)) \
            if kappa > 0 else math.sqrt(t)
        sd = sigma0 * VOL_SCALE * var_f
        teff = (1.0 - decay) / kappa if kappa > 0 else t
        for _ in range(N_DRAWS):
            scored = []
            for c in cands:
                s = survival(c["f"][4], t, surv)
                if s < 1.0 and rng.random() > s:
                    continue
                vel = ridge_velocity(beta, c["f"])
                v_eq = a + b * c["f"][0]
                dv = vel - v_eq
                if is_breakout(c["f"]):
                    dv *= (1.0 + BREAKOUT_EXTRA)
                dv *= DRIFT_SCALE
                mean = xbar + (c["f"][0] - xbar) * decay + dv * teff
                mean = min(max(mean, -0.3), 4.9)
                x = mean + sd * rng.gauss(0.0, 1.0)
                scored.append((x, c["uid"]))
            scored.sort(key=lambda p: (p[0], p[1]))
            for place in range(1, 11):
                if place - 1 < len(scored):
                    freq[scored[place - 1][1]][place] += 1
            for pos, (_, uid) in enumerate(scored[:10]):
                rank_sum[uid] += pos + 1
                rank_cnt[uid] += 1
            n_sims += 1
    M = {}
    mean_rank = {}
    for c in cands:
        uid = c["uid"]
        M[uid] = [0.0] + [freq[uid][k] / n_sims for k in range(1, 11)]
        mean_rank[uid] = rank_sum[uid] / rank_cnt[uid] if rank_cnt[uid] else 99.0
    return M, mean_rank


def survival(age_h, t, surv):
    """Measured survival: P(a game of chart-age age_h is still on the
    chart after t hours). Falls back to a smooth prior when unmeasured."""
    ab = 0 if age_h < 72 else (1 if age_h < 168 else (2 if age_h < 336 else 3))
    b = bin_index(t)
    if b is None:
        b = 0 if t <= 0 else len(BINS) - 1
    v = surv.get((ab, b))
    if v is not None:
        return v
    # fall back to the nearest measured bin for this age bucket
    for bb in range(len(BINS) - 1, -1, -1):
        v = surv.get((ab, bb))
        if v is not None:
            return v
    # smooth prior: older games leave faster, longer horizons worse
    return min(max(0.97 - 0.004 * t - 0.0004 * age_h, 0.3), 0.99)

def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, reads = load_input(input_dir)
    places = int(task.get("places", 10))
    window = task["window"]
    wfrom = parse_time(window["from"])
    wuntil = parse_time(window["until"])
    cut_time = parse_time(task.get("history_until", window["from"]))
    if not reads:
        raise SystemExit("no reads in history.ndjson")
    charts = [chart_of(r) for r in reads]
    times = [parse_time(r["at"]) for r in reads]
    traj = build_trajectories(charts, times)

    # per-read feature vectors (reads <= i only)
    feats = []
    for i in range(len(reads)):
        fi = {}
        for uid in charts[i]:
            f = features_at(uid, traj, i, times)
            if f is not None:
                fi[uid] = f
        feats.append(fi)

    origins, newc, surv = build_origins(charts, times, feats, cut_time)
    beta, sigma0, a, b = fit_ridge(charts, times, feats, cut_time)
    kappa = min(max(-b, 0.002), 0.05)
    xbar = -a / b if b < -0.0005 else 3.5
    xbar = min(max(xbar, 2.0), 4.5)

    last = charts[-1]
    last_time = times[-1]
    cands = []
    for uid in sorted(last):
        f = feats[-1].get(uid)
        if f is None:
            continue
        cands.append({"uid": uid, "f": f})
    if not cands:
        raise SystemExit("empty chart at last read")

    # slot times (hours after window start), evenly spaced
    total_hours = (wuntil - wfrom).total_seconds() / 3600.0
    slot_hs = [(i + 0.5) * total_hours / N_SLOTS for i in range(N_SLOTS)]
    base_h = max((wfrom - last_time).total_seconds() / 3600.0, 0.0)
    # bin weights: fraction of the 72 scored slots landing in each bin
    slot_bins = [bin_index(base_h + sh) for sh in
                 [(i + 0.5) * total_hours / 72.0 for i in range(72)]]
    slot_bins = [s if s is not None else len(BINS) - 1 for s in slot_bins]
    bin_w = [slot_bins.count(bb) / 72.0 for bb in range(len(BINS))]

    # --- kernel component ---
    kern = {}
    for c in cands:
        K, ess = kernel_places(c["f"], origins, cut_time)
        kern[c["uid"]] = (K, ess)

    # --- parametric (ridge OU MC) component ---
    rng = random.Random(SEED)
    if beta is None:
        M = {c["uid"]: [0.0] * 11 for c in cands}
        mean_rank = {c["uid"]: c["f"][0] * 10.0 for c in cands}
    else:
        M, mean_rank = mc_places(cands, beta, a, b, kappa, xbar, sigma0,
                                 surv, slot_hs, base_h, last_time, rng)

    # --- learned newcomer reserve per place (bin-weighted, smoothed) ---
    # global place profile of newcomers (share of newcomer slots by place)
    prof = [0.0] * 11
    ptot = 0.0
    for bb in range(len(BINS)):
        for k in range(1, 11):
            prof[k] += newc[bb][k]
            ptot += newc[bb][k]
    if ptot > 0:
        prof = [p / ptot for p in prof]
    else:
        prof = [0.0] + [0.1] * 10
    reserve = [0.0] * 11
    for k in range(1, 11):
        raw = sum(bin_w[bb] * newc[bb][k] for bb in range(len(BINS)))
        bin_tot = sum(bin_w[bb] * sum(newc[bb][1:]) / 10.0
                      for bb in range(len(BINS)))
        smooth = bin_tot * prof[k] * 10.0
        r = 0.6 * raw + 0.4 * smooth
        reserve[k] = min(max(r * RESERVE_SCALE, RESERVE_FLOOR), 0.45)

    # --- blend + calibration ---
    probabilities = []
    exp_rank = {}
    for c in cands:
        uid = c["uid"]
        K, ess = kern[uid]
        p_place = [0.0] * 11
        for k in range(1, 11):
            kv = sum(bin_w[bb] * K[bb][k] for bb in range(len(BINS)))
            mv = M[uid][k]
            ess_avg = sum(bin_w[bb] * ess[bb] for bb in range(len(BINS)))
            alpha = ALPHA * ess_avg / (ess_avg + ESS0) if beta is not None \
                else 1.0
            p_place[k] = alpha * kv + (1.0 - alpha) * mv
        exp_rank[uid] = sum(k * p_place[k] for k in range(1, 11)) \
            + 11.0 * (1.0 - sum(p_place[1:]))
        c["p"] = p_place
    for k in range(1, 11):
        # floor, then soft cap, then scale to (1 - reserve)
        vals = [(c["p"][k], c["uid"]) for c in cands if c["p"][k] > 0]
        if not vals:
            continue
        vals = [(max(p, FLOOR), uid) for p, uid in vals]
        capped = []
        removed = 0.0
        for p, uid in vals:
            if p > CAP:
                removed += (p - CAP) * (1.0 - CAP_SHRINK)
                capped.append((CAP + (p - CAP) * CAP_SHRINK, uid))
            else:
                capped.append((p, uid))
        if removed > 0:
            base = sum(p for p, _ in capped)
            if base > 0:
                capped = [(p + removed * p / base, uid) for p, uid in capped]
        tot = sum(p for p, _ in capped)
        target = 1.0 - reserve[k]
        scale = target / tot if tot > 0 else 0.0
        rows = sorted(((p * scale, uid) for p, uid in capped),
                      key=lambda r: (-r[0], r[1]))[:100]
        for p, uid in rows:
            if p >= 0.0005:
                probabilities.append({"universe_id": int(uid), "place": k,
                                      "p": round(float(p), 6)})
    top10 = [int(uid) for uid, _ in
             sorted(exp_rank.items(), key=lambda kv: (kv[1], kv[0]))[:places]]
    if len(top10) < places:
        seen = set(top10)
        for c in sorted(cands, key=lambda c: c["f"][0]):
            if c["uid"] not in seen:
                top10.append(int(c["uid"]))
                if len(top10) == places:
                    break

    out = {"probabilities": probabilities, "top10": top10}
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")
    print(f"wrote {out_path}: {len(probabilities)} prob rows, top10={top10}")
    print(f"fitted: kappa={kappa:.5f} xbar={xbar:.2f} sigma0={sigma0:.3f} "
          f"drift a={a:.4f} b={b:.5f} reserve={[round(r, 2) for r in reserve[1:]]}")


if __name__ == "__main__":
    main()
