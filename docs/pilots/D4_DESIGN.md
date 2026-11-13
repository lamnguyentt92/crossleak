# D4 (DP-Prototype) design pilot — ELD 2013, before the main freeze

Purpose: choose the prototype representation and the number of prototypes P before the main pre-registration.
Data: ELD **2013 only** (the attacker-training period; the main study evaluates utility on 2014). Cold-start targets
(history 24 h) from a 30 % subset of the victim pool; co-tenants and prototype contributors are the other victim-pool
users plus the aux pool (229 users a day); public centres from the public pool on 40 random 2013 days. Code:
`scripts/pilot_d4_design.py` (seed 9001); raw outputs: `d4_design_c2base.json`, `d4_design_c2small.json`.
Rule fixed before running: pick the representation and P with the largest gain at ε = 4 on the primary model (c2base).

## Result (gain vs isolated forecasting; retained = gain / gain of cross-learning with 15 real co-tenants)
c2base, 120 targets (728 s on 2 sandbox cores):

| Representation | P | ε = ∞ | ε = 4 | ε = 1 |
|---|---|---|---|---|
| dayfold (24-h profile, tiled) | 8 | +6.1 % (1.23) | **+5.5 % (1.10)** | +5.0 % (1.01) |
| dayfold | 4 | +5.5 % (1.11) | +4.9 % (0.98) | +4.7 % (0.94) |
| weekfold (168 h) | 4 | +3.3 % (0.67) | +4.7 % (0.95) | +3.4 % (0.68) |
| weekfold | 8 | +4.2 % (0.84) | +4.1 % (0.83) | +3.0 % (0.61) |
| lastweek (168 h) | 8 | +4.6 % (0.93) | +4.4 % (0.88) | +2.1 % (0.42) |
| full (336 h) | 8 | +4.3 % (0.86) | +2.6 % (0.52) | +0.7 % (0.14) |
| cross-learning, 15 co-tenants | – | +5.0 % (1.00) | | |

c2small (80 targets): cross-learning **hurt** (−3.5 %) and every D4 variant was between −0.2 % and −10 %; the small model
does not benefit from co-tenants in this setting, so it cannot rank designs.

## Decision (DECISIONS.md D-018)
Representation **dayfold**, primary **P = 8**; P = 4 kept as an estimand. Interpretation (to be tested, not claimed): the
cold-start benefit comes from a population-level daily-shape prior; a 24-dimensional representation has the lowest
sensitivity per unit of signal, so it survives DP noise best. 120 targets carry no confidence interval; this pilot only
fixes the design.
