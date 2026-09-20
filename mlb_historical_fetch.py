#!/usr/bin/env python3
"""
mlb_historical_fetch.py  (slim edition, schema v1)

Same job and same command-line flags as before, but every response is reduced
to only the fields a website needs BEFORE it is written to disk. Nothing raw is
saved. See SCHEMA.md for the exact format of each file.

What changed vs. the raw version
  - 4 files per game (linescore, boxscore, playbyplay, winprobability)
    -> 1 file per game:   seasons/{YEAR}/games/{gamePk}.json
  - 30 roster files per season -> 1 file: seasons/{YEAR}/rosters.json
  - draft, transactions, standings-splits: only displayed fields kept
  - all JSON is minified (no indentation)
  - transient network failures are NOT saved as empty files, so the next run
    retries them (before, a failed request was saved as {} and skipped forever)
  - postponed / cancelled games are skipped (they have no box score)
  - files are written atomically (temp file, then rename)

Output layout:
    data/
      seasons/{YEAR}/
        games/{gamePk}.json
        rosters.json
        transactions/{01..12}.json   (one file per month)
        standings-splits.json
      draft/{YEAR}.json
      coverage_report_*.json / .csv

Usage:
    python3 mlb_historical_fetch.py --only-years 2019 --games-per-year-limit 5
    python3 mlb_historical_fetch.py --start 1980 --end 1989 --output-dir data
    python3 mlb_historical_fetch.py --date 2026-09-19        # daily mode
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

BASE = "https://statsapi.mlb.com/api/v1"
SPORT_ID = 1  # MLB
REQUEST_DELAY = 0.6
MAX_RETRIES = 4
RETRY_BACKOFF = 2.0
SCHEMA_VERSION = 1

START_YEAR = 1980
END_YEAR = 2025

SKIP_GAME_STATES = {"Postponed", "Cancelled"}

DATA_DIR = Path("data")
COVERAGE_SUFFIX = ""

# Returned by fetch_json when a request still failed after all retries.
# Different from None, which means "the API has nothing for this id (404/empty)".
FAILED = object()
FAILURES = 0   # number of requests that failed after all retries in this run


# --------------------------------------------------------------------------
# Networking + file helpers
# --------------------------------------------------------------------------

def fetch_json(url, retries=MAX_RETRIES):
    """GET url. Returns parsed JSON, None (404 / empty body) or FAILED."""
    last_err = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "personal-site-ingest/2.0"})
            with urlopen(req, timeout=30) as resp:
                raw = resp.read()
            time.sleep(REQUEST_DELAY)
            if not raw:
                return None
            return json.loads(raw)
        except HTTPError as e:
            if e.code == 404:
                return None
            last_err = e
        except (URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
        wait = RETRY_BACKOFF ** attempt
        print(f"    retry {attempt+1}/{retries} for {url} ({last_err}) - waiting {wait:.0f}s", file=sys.stderr)
        time.sleep(wait)
    print(f"    GAVE UP on {url}: {last_err}", file=sys.stderr)
    global FAILURES
    FAILURES += 1
    return FAILED


def save_json(path: Path, data, pretty=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def g(d, *keys, default=None):
    """Safe nested get: g(x, 'a', 'b') -> x['a']['b'] or default."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d


def is_meaningfully_empty(data):
    if data is None or data is FAILED:
        return True
    if isinstance(data, dict):
        if not data:
            return True
        if set(data.keys()) <= {"copyright"}:
            return True
    if isinstance(data, list) and not data:
        return True
    return False


class Coverage:
    def __init__(self):
        self.rows = {}

    def record(self, year, field, had_data: bool):
        row = self.rows.setdefault((year, field), {"total": 0, "nonempty": 0})
        row["total"] += 1
        if had_data:
            row["nonempty"] += 1

    def write(self):
        out = []
        for (year, field), row in sorted(self.rows.items()):
            pct = (row["nonempty"] / row["total"] * 100) if row["total"] else 0.0
            out.append({"year": year, "field": field, "total_checked": row["total"],
                        "had_data": row["nonempty"], "pct_available": round(pct, 1)})
        json_path = DATA_DIR / f"coverage_report{COVERAGE_SUFFIX}.json"
        csv_path = DATA_DIR / f"coverage_report{COVERAGE_SUFFIX}.csv"
        save_json(json_path, out, pretty=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["year", "field", "total_checked", "had_data", "pct_available"])
            w.writeheader()
            w.writerows(out)
        print(f"\nCoverage report written: {json_path} / {csv_path} ({len(out)} rows)")


# --------------------------------------------------------------------------
# Slimmers: raw API response -> compact structure (see SCHEMA.md)
# --------------------------------------------------------------------------

BAT_FIELDS = [("ab", "atBats"), ("r", "runs"), ("h", "hits"), ("d", "doubles"),
              ("t", "triples"), ("hr", "homeRuns"), ("rbi", "rbi"), ("bb", "baseOnBalls"),
              ("k", "strikeOuts"), ("sb", "stolenBases"), ("cs", "caughtStealing"),
              ("lob", "leftOnBase"), ("hbp", "hitByPitch"), ("sf", "sacFlies"),
              ("sh", "sacBunts")]

PIT_FIELDS = [("ip", "inningsPitched"), ("h", "hits"), ("r", "runs"), ("er", "earnedRuns"),
              ("bb", "baseOnBalls"), ("k", "strikeOuts"), ("hr", "homeRuns"),
              ("np", "numberOfPitches"), ("bf", "battersFaced")]


def _pick(stats, spec):
    out = {}
    if not isinstance(stats, dict):
        return out
    for short, long in spec:
        v = stats.get(long)
        if v is not None:
            out[short] = v
    return out


def slim_linescore(ls):
    if is_meaningfully_empty(ls):
        return None
    inn = []
    for i in ls.get("innings") or []:
        row = []
        for side in ("away", "home"):
            s = i.get(side) or {}
            row += [s.get("runs"), s.get("hits"), s.get("errors"), s.get("leftOnBase")]
        inn.append(row)

    def tot(side):
        t = g(ls, "teams", side, default={})
        return [t.get("runs"), t.get("hits"), t.get("errors"), t.get("leftOnBase")]

    return {"inn": inn, "a": tot("away"), "h": tot("home"), "si": ls.get("scheduledInnings")}


def _slim_box_team(t, names):
    players = t.get("players") or {}
    by_id = {}
    for p in players.values():
        pid = g(p, "person", "id")
        if pid is not None:
            by_id[pid] = p
            full = g(p, "person", "fullName")
            if full:
                names[str(pid)] = full

    bat = []
    for pid in t.get("batters") or []:
        p = by_id.get(pid)
        if not p:
            continue
        line = {"id": pid, "n": g(p, "person", "boxscoreName"),
                "pos": g(p, "position", "abbreviation"), "bo": p.get("battingOrder")}
        line.update(_pick(g(p, "stats", "batting", default={}), BAT_FIELDS))
        bat.append(line)
    bat.sort(key=lambda x: int(x["bo"]) if str(x.get("bo") or "").isdigit() else 9999)

    pit = []
    for pid in t.get("pitchers") or []:
        p = by_id.get(pid)
        if not p:
            continue
        ps = g(p, "stats", "pitching", default={})
        line = {"id": pid, "n": g(p, "person", "boxscoreName")}
        line.update(_pick(ps, PIT_FIELDS))
        if ps.get("note"):
            line["note"] = ps["note"]
        pit.append(line)

    notes = {}
    for sec in t.get("info") or []:
        rows = [[f.get("label"), f.get("value")] for f in sec.get("fieldList") or []]
        if rows and sec.get("title"):
            notes[sec["title"]] = rows

    team = t.get("team") or {}
    out = {"id": team.get("id"), "name": team.get("name"), "ab": team.get("abbreviation"),
           "bat": bat, "pit": pit,
           "tot": _pick(g(t, "teamStats", "batting", default={}), BAT_FIELDS)}
    if notes:
        out["notes"] = notes
    return out


def slim_boxscore(bx, names):
    if is_meaningfully_empty(bx):
        return None
    teams = bx.get("teams") or {}
    box = {"a": _slim_box_team(teams.get("away") or {}, names),
           "h": _slim_box_team(teams.get("home") or {}, names)}
    info = [[i.get("label"), i.get("value", "")] for i in bx.get("info") or []]
    if info:
        box["info"] = info
    off = [[o.get("officialType"), g(o, "official", "fullName")] for o in bx.get("officials") or []]
    if off:
        box["off"] = off
    venue = g(teams, "home", "team", "venue", default=None)
    if venue:
        box["venue"] = {"id": venue.get("id"), "name": venue.get("name")}
    return box


def _slim_pitch(ev):
    d = ev.get("details") or {}
    pdata = ev.get("pitchData") or {}
    coords = pdata.get("coordinates") or {}

    def r2(x):
        return round(x, 2) if isinstance(x, (int, float)) else None

    row = [d.get("code") or "", g(d, "type", "code"), pdata.get("startSpeed"),
           pdata.get("zone"), r2(coords.get("pX")), r2(coords.get("pZ"))]
    while len(row) > 1 and row[-1] is None:
        row.pop()
    return row


def slim_playbyplay(pbp, names):
    if is_meaningfully_empty(pbp):
        return None
    plays = []
    for p in pbp.get("allPlays") or []:
        res = p.get("result") or {}
        about = p.get("about") or {}
        cnt = p.get("count") or {}
        m = p.get("matchup") or {}
        bid, pid = g(m, "batter", "id"), g(m, "pitcher", "id")
        for who in ("batter", "pitcher"):
            i, n = g(m, who, "id"), g(m, who, "fullName")
            if i is not None and n and str(i) not in names:
                names[str(i)] = n

        row = {"i": about.get("atBatIndex"), "in": about.get("inning"),
               "t": 0 if about.get("isTopInning") else 1,
               "o": cnt.get("outs"), "b": cnt.get("balls"), "s": cnt.get("strikes"),
               "as": res.get("awayScore"), "hs": res.get("homeScore"),
               "ev": res.get("event"), "et": res.get("eventType"), "d": res.get("description"),
               "bt": bid, "p": pid, "bs": g(m, "batSide", "code"), "ph": g(m, "pitchHand", "code")}
        if res.get("rbi"):
            row["rbi"] = res["rbi"]
        if about.get("isScoringPlay"):
            row["sc"] = 1

        pitches, actions, hit = [], [], None
        for ev in p.get("playEvents") or []:
            if ev.get("isPitch"):
                pitches.append(_slim_pitch(ev))
            elif ev.get("type") == "action":
                desc = g(ev, "details", "description")
                if desc:
                    actions.append(desc)
            hd = ev.get("hitData")
            if hd:
                hit = [hd.get("launchSpeed"), hd.get("launchAngle"),
                       hd.get("totalDistance"), hd.get("trajectory")]
                while hit and hit[-1] in (None, ""):
                    hit.pop()
        if pitches:
            row["pt"] = pitches
        if actions:
            row["ac"] = actions
        if hit:
            row["hd"] = hit
        plays.append(row)
    return plays


def slim_winprob(wp):
    if not isinstance(wp, list) or not wp:
        return None
    out = []
    for it in wp:
        idx = it.get("atBatIndex")
        if idx is None:
            idx = g(it, "about", "atBatIndex")
        home = it.get("homeTeamWinProbability")
        add = it.get("homeTeamWinProbabilityAdded")
        out.append([idx,
                    round(home, 1) if isinstance(home, (int, float)) else None,
                    round(add, 1) if isinstance(add, (int, float)) else None])
    return out


def build_game(pk, date, ls, bx, pbp, wp):
    names = {}
    box = slim_boxscore(bx, names)
    plays = slim_playbyplay(pbp, names)
    return {"v": SCHEMA_VERSION, "pk": pk, "date": date,
            "ls": slim_linescore(ls), "box": box, "plays": plays,
            "wp": slim_winprob(wp), "names": names}


def slim_roster(data):
    rows = []
    for r in (data or {}).get("roster") or []:
        rows.append([g(r, "person", "id"), g(r, "person", "fullName"), r.get("jerseyNumber") or "",
                     g(r, "position", "abbreviation"), g(r, "status", "code")])
    return rows


def slim_draft(data, year):
    picks = []
    for rnd in g(data, "drafts", "rounds", default=[]):
        for pk in rnd.get("picks") or []:
            person = pk.get("person") or {}
            home = pk.get("home") or {}
            hm = ", ".join(x for x in (home.get("city"), home.get("state"), home.get("country")) if x)
            picks.append([pk.get("pickRound"), pk.get("pickNumber"), g(pk, "team", "id"),
                          person.get("id"), person.get("fullName"),
                          g(person, "primaryPosition", "abbreviation"),
                          g(pk, "school", "name"), person.get("birthDate"),
                          person.get("height"), person.get("weight"),
                          g(person, "batSide", "code"), g(person, "pitchHand", "code"), hm])
    return {"v": SCHEMA_VERSION, "year": year, "picks": picks}


def slim_transactions_by_month(data, year):
    """Returns {"01": {...}, ..., "12": {...}} (plus "00" for rows with no date)."""
    months = {f"{m:02d}": {"types": {}, "tx": []} for m in range(1, 13)}
    for t in (data or {}).get("transactions") or []:
        date = t.get("date") or t.get("effectiveDate") or ""
        mm = date[5:7] if len(date) >= 7 and date[5:7].isdigit() else "00"
        bucket = months.setdefault(mm, {"types": {}, "tx": []})
        code = t.get("typeCode")
        if code and t.get("typeDesc"):
            bucket["types"][code] = t["typeDesc"]
        bucket["tx"].append([date, code, g(t, "person", "id"), g(t, "person", "fullName"),
                             g(t, "fromTeam", "id"), g(t, "toTeam", "id"), t.get("description")])
    out = {}
    for mm, b in months.items():
        if mm == "00" and not b["tx"]:
            continue
        out[mm] = {"v": SCHEMA_VERSION, "year": year, "month": mm, "types": b["types"], "tx": b["tx"]}
    return out


def _split_map(records):
    out = {}
    for r in records or []:
        key = r.get("type")
        if key is not None:
            out[key] = [r.get("wins"), r.get("losses")]
    return out


def slim_standings(data, year):
    teams, updated = [], None
    for rec in (data or {}).get("records") or []:
        updated = updated or rec.get("lastUpdated")
        for tr in rec.get("teamRecords") or []:
            t = tr.get("team") or {}
            recs = tr.get("records") or {}
            teams.append({
                "id": t.get("id"), "n": t.get("name"), "ab": t.get("abbreviation"),
                "lg": g(t, "league", "id", default=g(rec, "league", "id")),
                "dv": g(t, "division", "id", default=g(rec, "division", "id")),
                "w": tr.get("wins"), "l": tr.get("losses"), "pct": tr.get("winningPercentage"),
                "gp": tr.get("gamesPlayed"), "rs": tr.get("runsScored"), "ra": tr.get("runsAllowed"),
                "rd": tr.get("runDifferential"), "dr": tr.get("divisionRank"),
                "lr": tr.get("leagueRank"), "gb": tr.get("gamesBack"),
                "wcgb": tr.get("wildCardGamesBack"), "cl": tr.get("clinchIndicator"),
                "st": g(tr, "streak", "streakCode"),
                "sp": _split_map(recs.get("splitRecords")),
                "ex": _split_map(recs.get("expectedRecords")),
            })
    return {"v": SCHEMA_VERSION, "year": year, "updated": updated, "teams": teams}


# --------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------

def get_season_games(year, limit=None):
    """[(gamePk, 'YYYY-MM-DD'), ...] for a season; postponed/cancelled excluded."""
    url = (f"{BASE}/schedule?sportId={SPORT_ID}&season={year}"
           f"&gameType=R,F,D,L,W,C,P,A"
           f"&fields=dates,date,games,gamePk,status,detailedState")
    data = fetch_json(url)
    if data is FAILED:
        print(f"  WARNING: schedule for {year} failed - rerun to retry this year", file=sys.stderr)
        return []
    return _games_from_schedule(data, limit)


def _games_from_schedule(data, limit=None):
    games = []
    for de in (data or {}).get("dates", []):
        for gm in de.get("games", []):
            if g(gm, "status", "detailedState") in SKIP_GAME_STATES:
                continue
            games.append((gm["gamePk"], de.get("date")))
    return games[:limit] if limit else games


def get_date_games(date_str):
    url = (f"{BASE}/schedule?sportId={SPORT_ID}&date={date_str}"
           f"&fields=dates,date,games,gamePk,status,detailedState")
    data = fetch_json(url)
    if data is FAILED:
        return []
    return _games_from_schedule(data)


def fetch_game_data(year, game_pk, date, skip_existing, coverage):
    path = DATA_DIR / "seasons" / str(year) / "games" / f"{game_pk}.json"
    if skip_existing and path.exists():
        ex = load_json(path)
        if ex:
            for field, key in (("linescore", "ls"), ("boxscore", "box"),
                               ("playbyplay", "plays"), ("winprobability", "wp")):
                coverage.record(year, field, bool(ex.get(key)))
        return

    urls = [("linescore", f"{BASE}/game/{game_pk}/linescore"),
            ("boxscore", f"{BASE}/game/{game_pk}/boxscore"),
            ("playbyplay", f"{BASE}/game/{game_pk}/playByPlay"),
            ("winprobability", f"{BASE}/game/{game_pk}/winProbability")]
    raw = {}
    for field, url in urls:
        raw[field] = fetch_json(url)

    if any(v is FAILED for v in raw.values()):
        print(f"    game {game_pk}: a request failed - not saved, will retry next run", file=sys.stderr)
        return

    for field in raw:
        coverage.record(year, field, not is_meaningfully_empty(raw[field]))
    game = build_game(game_pk, date, raw["linescore"], raw["boxscore"],
                      raw["playbyplay"], raw["winprobability"])
    save_json(path, game)


def fetch_team_ids():
    data = fetch_json(f"{BASE}/teams?sportId={SPORT_ID}")
    if not data or data is FAILED:
        return []
    return [t["id"] for t in data.get("teams", [])]


def fetch_rosters(year, team_ids, skip_existing, coverage):
    path = DATA_DIR / "seasons" / str(year) / "rosters.json"
    if skip_existing and path.exists():
        ex = load_json(path)
        if ex:
            for rows in (ex.get("teams") or {}).values():
                coverage.record(year, "roster", bool(rows))
        return
    teams = {}
    for team_id in team_ids:
        data = fetch_json(f"{BASE}/teams/{team_id}/roster?rosterType=fullSeason&season={year}")
        if data is FAILED:
            print(f"    roster {team_id}/{year} failed - rosters.json not saved, rerun to retry", file=sys.stderr)
            return
        rows = slim_roster(data)
        coverage.record(year, "roster", bool(rows))
        if rows:
            teams[str(team_id)] = rows
    save_json(path, {"v": SCHEMA_VERSION, "year": year, "teams": teams})


def _simple_fetch(path, url, slimmer, body_key, year, field, skip_existing, coverage):
    if skip_existing and path.exists():
        ex = load_json(path)
        if ex is not None:
            coverage.record(year, field, bool(ex.get(body_key)))
        return
    data = fetch_json(url)
    if data is FAILED:
        print(f"    {field} {year} failed - not saved, rerun to retry", file=sys.stderr)
        return
    slim = slimmer(data, year)
    coverage.record(year, field, not is_meaningfully_empty(data))
    save_json(path, slim)


def fetch_draft(year, skip_existing, coverage):
    _simple_fetch(DATA_DIR / "draft" / f"{year}.json", f"{BASE}/draft/{year}",
                  slim_draft, "picks", year, "draft", skip_existing, coverage)


def fetch_transactions(year, skip_existing, coverage):
    folder = DATA_DIR / "seasons" / str(year) / "transactions"
    marker = folder / "12.json"          # written last, so its presence means "complete"
    if skip_existing and marker.exists():
        n = 0
        for f in folder.glob("*.json"):
            ex = load_json(f)
            n += len((ex or {}).get("tx") or [])
        coverage.record(year, "transactions", n > 0)
        return
    url = f"{BASE}/transactions?startDate={year}-01-01&endDate={year}-12-31&sportId={SPORT_ID}"
    data = fetch_json(url)
    if data is FAILED:
        print(f"    transactions {year} failed - not saved, rerun to retry", file=sys.stderr)
        return
    coverage.record(year, "transactions", not is_meaningfully_empty(data) and bool(data.get("transactions")))
    for mm, payload in sorted(slim_transactions_by_month(data, year).items()):
        save_json(folder / f"{mm}.json", payload)


def fetch_standings_splits(year, skip_existing, coverage):
    url = (f"{BASE}/standings?leagueId=103,104&season={year}"
           f"&standingsTypes=regularSeason&hydrate=team")
    _simple_fetch(DATA_DIR / "seasons" / str(year) / "standings-splits.json", url,
                  slim_standings, "teams", year, "standings_splits", skip_existing, coverage)


def fetch_single_date(date_str, coverage, games_only=False):
    """Daily mode: this date's games + force-refresh of the year-level files
    (draft, transactions, standings, rosters) unless games_only is set."""
    year = int(date_str[:4])
    print(f"=== {date_str} (year {year}) ===")
    if not games_only:
        fetch_draft(year, False, coverage)
        fetch_transactions(year, False, coverage)
        fetch_standings_splits(year, False, coverage)
        fetch_rosters(year, fetch_team_ids(), False, coverage)
    games = get_date_games(date_str)
    print(f"  {len(games)} games scheduled on {date_str}...")
    for pk, d in games:
        fetch_game_data(year, pk, d or date_str, False, coverage)
    print(f"  {date_str} complete.\n")


def finish():
    """Exit code 2 if any request failed, so workflows know the run was incomplete."""
    if FAILURES:
        print(f"\n{FAILURES} request(s) failed - data is incomplete. Rerun to fill the gaps.", file=sys.stderr)
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=int, default=START_YEAR)
    parser.add_argument("--end", type=int, default=END_YEAR)
    parser.add_argument("--only-years", type=int, nargs="*", default=None)
    parser.add_argument("--date", type=str, default=None,
                        help="YYYY-MM-DD daily mode (overrides --start/--end/--only-years)")
    parser.add_argument("--games-only", action="store_true",
                        help="With --date: fetch only that day's games (skip draft/transactions/standings/rosters)")
    parser.add_argument("--games-per-year-limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    # Flush output line by line so GitHub Actions logs show progress live
    # (Python otherwise buffers stdout when it isn't a terminal).
    sys.stdout.reconfigure(line_buffering=True)

    global DATA_DIR, COVERAGE_SUFFIX
    DATA_DIR = Path(args.output_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.date:
        COVERAGE_SUFFIX = f"_{args.date}"
        coverage = Coverage()
        print(f"Output directory: {DATA_DIR.resolve()}")
        fetch_single_date(args.date, coverage, games_only=args.games_only)
        coverage.write()
        print("Date processed.")
        finish()
        return

    years = args.only_years if args.only_years else list(range(args.start, args.end + 1))
    COVERAGE_SUFFIX = f"_{min(years)}-{max(years)}"
    coverage = Coverage()
    print(f"Output directory: {DATA_DIR.resolve()}")
    print(f"Years this run: {min(years)}-{max(years)} ({len(years)} years)\n")

    team_ids = fetch_team_ids()
    print(f"{len(team_ids)} teams found.\n")

    for year in years:
        print(f"=== {year} ===")
        fetch_draft(year, args.skip_existing, coverage)
        fetch_transactions(year, args.skip_existing, coverage)
        fetch_standings_splits(year, args.skip_existing, coverage)
        fetch_rosters(year, team_ids, args.skip_existing, coverage)

        games = get_season_games(year, limit=args.games_per_year_limit)
        print(f"  {len(games)} games to process...")
        for i, (pk, d) in enumerate(games, 1):
            fetch_game_data(year, pk, d, args.skip_existing, coverage)
            if i % 50 == 0:
                print(f"    ...{i}/{len(games)} games done")
        print(f"  {year} complete.\n")
        coverage.write()

    print("All years processed.")
    coverage.write()
    finish()


if __name__ == "__main__":
    main()
