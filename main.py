

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt


# ---- appliance stuff -------------------------------------------------

@dataclass
class Appliance:
    name: str
    watts: float
    on_hours: set      # which hours (0-23) it's drawing power
    surge_mult: float = 1.0   # only matters for inverter surge sizing


def build_hourly_load(appliances, n_days):
    # stamping the same daily schedule across however many autonomy days
    # we're modeling. real households probably vary day to day but that's
    # a rabbit hole I don't have time for right now
    T = 24 * n_days
    L = np.zeros(T)
    for d in range(n_days):
        for a in appliances:
            for h in a.on_hours:
                L[d * 24 + h] += a.watts
    return L


def worst_case_surge(appliances):
    # everything switches on at the exact same second, worst case. probably
    # never actually happens in real life but better to oversize the
    # inverter a bit than have it trip every morning
    total = 0
    for a in appliances:
        total = total + (a.watts * a.surge_mult)
    return total


# ---- solar curve -------------------------------------------------

def solar_curve_clear_sky(sunrise=6, sunset=18):
    # sine bell, 0 before sunrise and after sunset. not pulling real
    # irradiance data for this (TODO, hook up NASA POWER api or smth at
    # some point) just hardcoding a clean curve for now
    curve = np.zeros(24)
    span = sunset - sunrise
    for h in range(24):
        if h >= sunrise and h < sunset:
            curve[h] = np.sin(np.pi * (h - sunrise) / span)
    return curve


def build_irradiance(n_days, weather_factors):
    # weather_factors like [1.0, 0.3] means day 1 full sun, day 2 mostly
    # cloudy - this is basically the whole point of "days of autonomy",
    # what happens if the sun doesn't really show up for a stretch
    base = solar_curve_clear_sky()
    I = np.zeros(24 * n_days)
    for d in range(n_days):
        I[d*24 : d*24+24] = base * weather_factors[d]
    return I


# ---- the actual LP -------------------------------------------------
#
# variable layout, one long vector:
#   [0] = x1 = panel watts
#   [1] = x2 = battery Wh
#   [2] = x3 = inverter watts
#   next T   = SOC_t
#   next T   = P_charge_t
#   next T   = P_discharge_t
#   next T   = P_direct_t   (solar straight to load, skips battery)
#
# doing index math by hand with helper lambdas instead of some variable
# manager class - felt like more trouble than it's worth for something
# this size

def solve_system(L, I, panel_cost, battery_cost, inverter_cost,
                  eta_ch=0.95, eta_dis=0.95, eta_sys=0.85, dod_max=0.8,
                  appliances=None):

    T = len(L)
    n = 3 + 4*T

    soc_i = lambda t: 3 + t
    ch_i = lambda t: 3 + T + t
    dis_i = lambda t: 3 + 2*T + t
    dir_i = lambda t: 3 + 3*T + t

    c = np.zeros(n)
    c[0] = panel_cost
    c[1] = battery_cost
    c[2] = inverter_cost
    # rest of the cost vector stays 0 - SOC/charge/discharge/direct don't
    # cost anything on their own, they're just there so the solver has to
    # prove the system actually works hour by hour

    # SOC recursion, equality constraints, one row per hour
    A_eq = np.zeros((T, n))
    b_eq = np.zeros(T)
    for t in range(T):
        A_eq[t, soc_i(t)] = 1
        A_eq[t, ch_i(t)] = -eta_ch
        A_eq[t, dis_i(t)] = 1/eta_dis
        if t == 0:
            # assuming battery starts full on day 1 - fair since that's
            # basically how you'd commission it anyway
            A_eq[t, 1] = -1
        else:
            A_eq[t, soc_i(t-1)] = -1

    ub_rows = []
    ub_rhs = []

    # gen limit: direct_t + ch_t <= x1 * I_t * eta_sys
    for t in range(T):
        row = np.zeros(n)
        row[dir_i(t)] = 1
        row[ch_i(t)] = 1
        row[0] = -I[t] * eta_sys
        ub_rows.append(row)
        ub_rhs.append(0)

    # load has to be covered: direct_t + dis_t >= L_t
    for t in range(T):
        row = np.zeros(n)
        row[dir_i(t)] = -1
        row[dis_i(t)] = -1
        ub_rows.append(row)
        ub_rhs.append(-L[t])

    # SOC ceiling: SOC_t <= x2
    for t in range(T):
        row = np.zeros(n)
        row[soc_i(t)] = 1
        row[1] = -1
        ub_rows.append(row)
        ub_rhs.append(0)

    # SOC floor (DOD limit): SOC_t >= (1-dod_max)*x2
    for t in range(T):
        row = np.zeros(n)
        row[soc_i(t)] = -1
        row[1] = (1 - dod_max)
        ub_rows.append(row)
        ub_rhs.append(0)

    # inverter needs to cover the single worst hour of load
    peak_load = 0
    for val in L:
        if val > peak_load:
            peak_load = val
    row = np.zeros(n)
    row[2] = -1
    ub_rows.append(row)
    ub_rhs.append(-peak_load)

    # surge headroom - most inverters can survive ~2x rated for a couple
    # seconds on motor startup, so continuous rating only needs to cover
    # half the surge number, not the whole thing. going off a few spec
    # sheets I looked at, not 100% sure this holds for every inverter type
    surge = worst_case_surge(appliances) if appliances else peak_load
    row = np.zeros(n)
    row[2] = -2.0
    ub_rows.append(row)
    ub_rhs.append(-surge)

    A_ub = np.array(ub_rows)
    b_ub = np.array(ub_rhs)

    bounds = [(0, None)] * n

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method="highs")

    if not res.success:
        # TODO: say which hour actually breaks feasibility instead of just
        # "nope, failed"
        raise RuntimeError("linprog couldn't solve this: " + res.message)

    x = res.x
    soc = x[3 : 3+T]
    pch = x[3+T : 3+2*T]
    pdis = x[3+2*T : 3+3*T]
    pdir = x[3+3*T : 3+4*T]

    # paranoid double check that the DOD floor actually held - should
    # already be guaranteed by the constraint but got burned once by a
    # sign error in the constraint matrix so now I always check by hand
    floor = (1 - dod_max) * x[1]
    for t in range(T):
        if soc[t] < floor - 0.5:  # tiny tolerance for solver rounding
            print(f"  (heads up: SOC at hour {t} is {soc[t]:.1f}, "
                  f"technically below the {floor:.1f} floor - probably just "
                  f"solver tolerance but flagging it anyway)")

    return {
        "panel_watts": x[0],
        "battery_wh": x[1],
        "inverter_watts": x[2],
        "cost_pkr": res.fun,
        "soc": soc,
        "p_charge": pch,
        "p_discharge": pdis,
        "p_direct": pdir,
        "surge_w": surge,
        "peak_load_w": peak_load,
    }


def plot_soc(soc, battery_wh, dod_max, n_days):
    T = len(soc)
    hrs = np.arange(T)
    floor = (1-dod_max) * battery_wh

    plt.figure(figsize=(10,4.5))
    plt.plot(hrs, soc, linewidth=2, label="SOC (Wh)")
    plt.axhline(battery_wh, linestyle="--", linewidth=1, label="full charge")
    plt.axhline(floor, linestyle="--", linewidth=1, color="red", label=f"DOD floor ({int(dod_max*100)}%)")

    for d in range(n_days):
        # shading night hours roughly so the plot's readable at a glance
        plt.axvspan(d*24+18, d*24+24, alpha=0.08, color="gray")
        plt.axvspan(d*24, d*24+6, alpha=0.08, color="gray")

    plt.xlabel("hour")
    plt.ylabel("battery SOC (Wh)")
    plt.title(f"SOC over {n_days} day(s) of autonomy")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("soc_profile.png", dpi=150)
    print("saved plot -> soc_profile.png")


def print_report(L, sizing, n_days, dod_max):
    print("="*60)
    print("micro-grid profiler - hourly sizing results")
    print("="*60)
    print(f"modeling {n_days} days (day 2+ assumed cloudy, worst case)")
    print(f"peak hourly load: {sizing['peak_load_w']:.0f} W")
    print(f"peak surge: {sizing['surge_w']:.0f} W")
    print()
    print("sized system:")
    print(f"  panel:    {sizing['panel_watts']:.1f} W")
    print(f"  battery:  {sizing['battery_wh']:.1f} Wh  (usable: {sizing['battery_wh']*dod_max:.1f} Wh @ {int(dod_max*100)}% DOD)")
    print(f"  inverter: {sizing['inverter_watts']:.1f} W")
    print(f"  cost:     PKR {sizing['cost_pkr']:,.0f}")
    print()
    print("SOC, first 24h (bar length is just for a quick visual, check the")
    print("plot for the actual numbers across all days):")
    for t in range(min(24, len(sizing["soc"]))):
        bl = int(sizing["soc"][t] / max(sizing["battery_wh"],1) * 30)
        print(f"  t={t:2d}  {sizing['soc'][t]:7.1f} Wh " + "#"*bl)
    print("="*60)


# ---- getting stuff from the user -------------------------------------------------
# figured most of the "hard science" constants (charge/discharge efficiency,
# DOD, the sunrise/sunset bell curve shape) aren't things a random person
# sizing their own system would actually know off the top of their head, so
# those stay as fixed assumptions up in solve_system(). what I AM asking for
# below is just the stuff someone actually knows without looking anything
# up - what appliances they've got, roughly what hours they run, how many
# bad-weather days they want to plan for, and rough prices.

def ask(prompt, default, cast=str):
    # little wrapper so I'm not rewriting the same try/except every time.
    # blank input just falls back to the default value
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"  (didn't understand that, using the default of {default})")
        return default


def parse_hour_range(text):
    # "19-4" should mean 7pm through 4am, wrapping past midnight. "12-21"
    # is just a normal same-day range. keeping the input format dead simple
    # since asking someone to type out a python set literal is a bit much
    start_s, end_s = text.split("-")
    start, end = int(start_s), int(end_s)
    if end > start:
        return set(range(start, end))
    else:
        return set(range(start, 24)) | set(range(0, end))


def get_appliances_from_user():
    print("\nenter your appliances one at a time. leave the name blank when you're done.\n")
    appliances = []
    while True:
        name = input("appliance name: ").strip()
        if name == "":
            break
        watts = ask("  wattage", 20, float)
        hrs_text = ask("  hours running (start-end, e.g. 19-4 means 7pm to 4am)", "18-22")
        try:
            on_hrs = parse_hour_range(hrs_text)
        except Exception:
            print("  (couldn't read that hour range, defaulting to 18-22)")
            on_hrs = set(range(18, 22))
        motor = ask("  got a motor that surges on startup, like a fan or pump? (y/n)", "n")
        surge = 3.0 if motor.lower().startswith("y") else 1.0
        appliances.append(Appliance(name, watts, on_hrs, surge_mult=surge))
        print()
    return appliances


DEMO_HOUSEHOLD = [
    Appliance("DC Fan 1", 30, set(range(12, 22)), surge_mult=3.0),
    Appliance("DC Fan 2", 30, set(range(12, 22)), surge_mult=3.0),
    Appliance("LED Bulb 1", 12, set(range(19, 24)) | set(range(0, 5))),
    Appliance("LED Bulb 2", 12, set(range(19, 24)) | set(range(0, 5))),
    Appliance("LED Bulb 3", 12, set(range(19, 24)) | set(range(0, 5))),
    Appliance("Mini Fridge (avg)", 45, set(range(24))),
]


if __name__ == "__main__":

    print("=== micro-grid profiler ===")
    mode = ask("run the demo household or enter your own appliances? (demo/custom)", "demo")

    if mode.lower().startswith("c"):
        household = get_appliances_from_user()
        if not household:
            print("nothing entered, falling back to the demo household instead")
            household = DEMO_HOUSEHOLD
    else:
        household = DEMO_HOUSEHOLD

    print()
    n_days = ask("days of autonomy (how many bad days in a row should the battery survive)", 2, int)
    worst_weather = ask("worst-day sun fraction, 0-1 (0.3 = a genuinely bad cloudy day)", 0.3, float)
    # day 1 assumed clear sky, everything after that uses the worst-case
    # weather number - could let someone set a different factor per day but
    # that's a lot of typing for not much extra insight
    weather = [1.0] + [worst_weather] * (n_days - 1) if n_days > 1 else [1.0]

    panel_cost = ask("panel cost, PKR per watt", 140, float)
    battery_cost = ask("battery cost, PKR per Wh", 55, float)
    inverter_cost = ask("inverter cost, PKR per watt", 35, float)

    L = build_hourly_load(household, n_days)
    I = build_irradiance(n_days, weather)

    sizing = solve_system(L, I, panel_cost, battery_cost, inverter_cost,
                           appliances=household)

    print()
    print_report(L, sizing, n_days, dod_max=0.8)
    plot_soc(sizing["soc"], sizing["battery_wh"], 0.8, n_days)
