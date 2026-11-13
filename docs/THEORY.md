# Theory: the privacy guarantee of DP-Prototype cross-learning (D4)

## Model
On day t the service holds the established users' contexts x_u (336 h ending at the origin). Each user maps its own
context to r_u = dayfold(normalise(x_u)) ∈ R^24 (normalise: zero mean, unit RMS over observed hours, missing → 0;
dayfold: mean over the 14 days per hour of day) and clips it: r̃_u = r_u · min(1, C / ||r_u||), C = √24. Public centres
c_1..c_P are fitted on public users only. User u belongs to cluster a(u) = argmin_p ||r̃_u − c_p|| (a function of u's own
data). The release M(X) = (S_p, n_p)_{p ≤ P} with S_p = Σ_{a(u)=p} r̃_u + N(0, σ_s² C² I_24), n_p = |{u: a(u)=p}| +
N(0, σ_n²). Prototypes π_p = S_p / n_p if n_p ≥ 1, else c_p. Every user v is forecast in the group {x_v} ∪ {π_p}.

## Proposition
With neighbouring datasets differing by one user (add/remove), M is ρ-zCDP with ρ = 1/(2σ_s²) + 1/(2σ_n²), hence
(ε, δ)-DP with ε = ρ + 2√(ρ ln(1/δ)). Every forecast returned on day t to any user or coalition is (ε, δ)-DP with respect
to every user outside the coalition.

## Proof
Adding user w changes only cluster a(w): S_{a(w)} by r̃_w (||r̃_w|| ≤ C) and n_{a(w)} by 1; all other coordinates of the
pre-noise release are unchanged (the assignment of other users does not depend on w; the centres are public). The
Gaussian mechanism with L2 sensitivity Δ and noise std σΔ is 1/(2σ²)-zCDP (Bun & Steinke 2016, Prop. 1.6); the sum and the
count are released by two Gaussian mechanisms (sensitivity C with noise σ_s C, sensitivity 1 with noise σ_n), so M is
ρ-zCDP by composition (Lemma 2.3). zCDP implies (ρ + 2√(ρ ln 1/δ), δ)-DP (Prop. 1.3). A coalition's forecasts are
functions of M(X), of the public model and centres, and of the coalition's own data, hence post-processing (Lemma 2.8). ∎

## Boundary cases (checked by tests or stated in the paper)
- Empty cluster or noisy count < 1: the public centre is used — a function of the noisy count only (post-processing).
- A user with no observed hour: r = 0, contributes 0 to the sum and 1 to the count; sensitivity bounds still hold.
- Clipping is not a no-op for adversarial inputs: an attacker's series is clipped like everyone's (test
  `test_release_sensitivity_add_one_user_changes_one_cluster` plants a norm-50 vector).
- Sybils: DP holds for any neighbouring pair, including datasets in which the attacker controls every other user; no
  assumption on group composition is needed (contrast with D2).
- Attribute inference: changing one user's record is remove + add, so the guarantee for inferring an attribute of a
  present user is (2ε, (1 + e^ε)δ); the DP audit compares empirical AUCs with that ceiling.
- Time: the unit of protection is one user's 14-day context in one daily release. A given hour of data appears in 14
  consecutive releases; protecting it costs 14ρ, and T days of releases cost Tρ (zCDP adds).
- The cold-start target does not contribute to the release it is forecast with (it has no 14-day history).

## Theory–code–experiment map
| Statement | Code | Test / experiment |
|---|---|---|
| Sensitivity of one user | `defenses/dp_prototype.py: clip_l2, release` | `tests/test_main.py` (sensitivity, clipping, negative control) |
| zCDP → (ε, δ) and noise scales | `eps_to_rho, rho_to_eps, noise_scales` | round-trip and monotonicity tests |
| Attribute ceiling | `dp_auc_ceiling, attribute_ceiling` | bounds test; DP audit (DPAUDIT rule) |
| Post-processing for every forecast | `experiments/main.py: Env.release` (one release per day, shared by all cells) | utility cells D4 |
