# Microgrid-Calculator
Sizing tool for off-grid solar setups, figures out the minimum panel capacity, battery capacity, and inverter capacity needed to keep a set of essential appliances running, at the lowest upfront cost . for the households I saw during the solar distribution drive, upfront capital is the actual constraint, not long-run efficiency.
how it works (math-wise)

linear program (scipy.optimize.linprog), minimizing:

cost = c1 * panel_watts + c2 * battery_wh + c3 * inverter_watts

subject to, for every hour t across however many days of autonomy you're modeling:

generation can't exceed what the panel + weather actually produces that hour
direct solar + battery discharge has to cover the load that hour
battery can't charge past 100% or discharge past the DOD floor
SOC this hour = SOC last hour + (charge in * efficiency) - (discharge out / efficiency)

plus two more just for the inverter:

has to handle the single worst hour of steady load
has to handle roughly half the "surge" wattage (motors pull way more current for a second on startup, inverters can usually survive 2x their rated continuous output for that brief a moment, going off a few spec sheets I looked at)

tracking SOC hour by hour instead of just balancing total daily energy is the important part - a system can look fine on a daily total and still have the battery die at 4am if the timing doesn't actually work out. hourly tracking catches that.

running it
pip install numpy scipy matplotlib
python3 microgrid_profiler.py

prints a text report + saves soc_profile.png (battery charge curve over the whole run, cloudy day included).

the demo case

2 DC fans (30W each, afternoon/evening), 3 LED bulbs (12W each, night), and a small fridge - included the fridge since that's honestly the thing most likely to actually break an off-grid system. lights and fans you can turn off if you're running low, a fridge full of food you kind of can't.

modeling 2 days: day 1 clear sky, day 2 assumed mostly cloudy (30% of normal output), to stress-test whether the battery survives a bad stretch of weather - which is really the whole point of "days of autonomy."

stuff that's still rough
solar curve is a hand-wavy sine bell between sunrise/sunset, not real weather data. would be better with actual hourly irradiance for Lahore (NASA POWER has this, just haven't hooked it up yet)
fridge is modeled as a flat continuous draw (averaged duty cycle) instead of the compressor actually cycling on and off. close enough for sizing, not physically accurate
appliance schedule repeats the exact same hours every day, real households aren't that regimented
costs are rough PKR/unit estimates, not real vendor quotes
the "2x surge = ok" assumption for the inverter isn't from anywhere authoritative, would want to confirm against whatever inverter actually ends up getting used
files
microgrid_profiler.py - everything, one file, runs standalone
soc_profile.png - generated when you run the script
