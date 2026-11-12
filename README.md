# roblox-up-and-coming-forecast (v2)

make.fun bounty entry: roblox-up-and-coming-forecast-01 — forecast the top 10 of Roblox's Up-and-Coming chart over the next 72 hours.

v2: learned kernel/analog forecaster + runtime ridge regression driving a corrected OU Monte Carlo; momentum 1.2× with a breakout filter; newcomer share learned per horizon bin; chart-age exit-hazard curve; calibration floors + soft cap.

`main.py` reads the chart history in $MAKEFUN_INPUT_DIR and writes forecast.json to $MAKEFUN_OUTPUT_DIR (deterministic, stdlib only).

License: MIT.
