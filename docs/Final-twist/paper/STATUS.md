# HAaNE I — paper status board

> **Living document — update IN PLACE** (rewrite stale rows; do not append logs). Read this to answer
> "where are we on paper I". Companions: `CLUSTER-RUNS-STATUS.md` (repo root, local-only; cluster
> state, last full update 2026-06-22), `CLAUDE.md` (resubmission direction + claims discipline),
> `../../../HAaNE-Resubmission-Plan-and-Proofs-2026-06-11.pdf` (the adjudicated plan + proofs),
> `../../../resubmission-artifacts-2026-06-11/` (drop-ins, tables, cover letter), and the series spec in
> the Neural-Networks-Matrices repo:
> `docs/superpowers/specs/2026-09-04-haane-series-boundary-and-imagenet-arm-design.md`.

**Paper:** `docs/Final-twist/paper/main.tex` (+ `sections/*.tex`, `tables/`, `figures/`; compiled `main.pdf`). Branch `refactor`, pushed; last commit `8b2f731` (2026-09-05); **not submitted**.
**Title (kept, decided 2026-09-04):** *Hidden Activations Are Not Enough I: Knowledge Matrices as Higher Representations.*
**Series (decided 2026-09-04):** HAaNE 0 = arXiv:2409.13163 (to be renamed); HAaNE I = this paper, a **continuation** of 0, its own paper — not a rewrite for resubmission; HAaNE II = the companion paper in Neural-Networks-Matrices. **Boundary rule:** I = the object at one trained network (x moves, θ fixed); II = the population statistic across training runs. I never uses "universality", "seed floor" or training controls; II never uses "higher representation".
**Target venue:** TMLR, submitted **before** II so II cites an arXiv id.

**Last updated:** 2026-09-05 — Plan C is **complete**: series edits C1–C4 (+ fix round 1), the ordering-table
regeneration C-s3 (+ fix rounds 1 and 2), the vocabulary appendix C5 and this board C6. All of it is **committed and pushed** (`8b2f731`, 2026-09-05):
that commit starts tracking the paper tree, which this repo's `.gitignore` had excluded, and stages the
carve-out so the repository explains its own state. The recipe it followed is
`…/Neural-Networks-Matrices/.claude/worktrees/learning-mechanics/.superpowers/sdd/2026-09-04-plan-C-haane-i-series-edits/commit-recipe.md`.

**Build (2026-09-05, `latexmk -pdf -interaction=nonstopmode main.tex`):** rc 0 · **0 errors** · 0 undefined
references or citations · 0 multiply-defined labels · **48 pages** · 23 overfull boxes (= the pre-Plan-C
baseline) · 3 underfull. Page history: 40 (C1–C4 + fix1) → 41 (C-s3 fix1, uncertainty + setup prose) →
42 (C-s3 fix2, the panel caption now enumerates all five attack-budget overrides) → **48** (C5, the
vocabulary appendix, +6). Layout: main matter §1–§10 pp. 2–31, Appendix A pp. 31–34, B 35–37, C 38–39,
**D (vocabulary) 40–45**, references 46–48.

**Series-vocabulary gates (all green 2026-09-05):** `universal|seed floor|training control` = **0** across
`sections/*.tex`, `main.tex`, `preamble.tex` *and* the new `sections/vocab/**`; `Part[~ -]I{1,2}` self-references
= **0** (the retargeting is done: the old "Part I"/"Part II" wording is gone everywhere — HAaNE II is the
companion paper on populations of training runs, and the applications paper is a separate, **unnumbered**
companion); `sec:A[1-6]` = 18; `tab:a-experiments` referenced twice; `\WARNING` = 0 in `tables/`, 4 in
`sections/` (all review-only, see the pre-submission checklist).

---

## Where we are (one screen)

**READY:** the theory core (11 theorem-level statements, all proved and numerically verified at float64 by two adversarial agents, 2026-06-11); Studies 1–3 and the honest negatives written; the Phase-1 nine-measure panel wired in (s1/s2 tables, study1/study4, S3 ordering appendix); reduce gate GREEN + controls `all_pass=True` (06-21); D2 KM-drift computed for DenseNet-121 and ResNet-152 (06-21) and then **dropped by decision (PATH B, Marco, 06-23)**: the fp32-vs-fp64 diagnostic (`14624489`, done 06-23) showed the measured `d_M` was a `load_cob_into_km` load bug (the teleported KM violates `M·1 = f` structurally; `tele_resid` unchanged at fp64), neither precision nor drift. Study 1b is reframed on what is solid — exact invariance (Thm maxinv), visible drift = logit gate (an equality), invisible part unestimated. **Remaining cluster jobs: NONE** (06-23 board; re-confirmed 2026-09-04).

**DONE since the 2026-09-04 board:**

- **Ordering tables regenerated from the reduces, with bootstrap intervals** (Task C-s3 + two fix rounds; script `scripts/regen_ordering_tables.py`, 1127 lines, 80 enforced self-checks, two runs byte-identical). `tab:ordering` **stays on the theorem45 reduce** — its six rows, the leave-one-out column and **Kendall $W = 0.921$** reproduce exactly, and `tab:coherence` reads the same reduce, so the headline and the coherence magnitudes cannot desynchronise (Ruling C-R8). The appendix panel `tables/s3_table.tex` is regenerated from the Phase-1 reduce `cluster-data/results/phase1/aggregated/s3_results.json` on a **population-matched** common set (n_common 512/1000/1000, because five ResNet-152 cells stored 1000 pairs and DeepFool only 512): trio **W = 0.9746, exact permutation p = 3.086e-05**, and the family grouping min{DeepFool, CW, Square} > FGSM > max{PGD, APGD} now holds on **all three** architectures. Every cell carries a seeded **BCa 95 % CI** and every adjacent-rank difference a percentile CI (B = 20,000); DenseNet-121 and GoogLeNet resolve every adjacent gap, ResNet-152's **top four do not** (DeepFool ≈ Square ≈ CW ≈ FGSM > PGD > APGD), which the panel prints as a point estimate with the resolution clause beside it. The three ordering `\WARNING` boxes and the "orderings disagree / trio W = 0.771" narrative are gone; review flag (1) is **RESOLVED** and flag (4) **PARTLY CLOSED**.
- **The sampling-order question closed in the paper's favour.** "The first 25,000 validation images by sorted filename" was challenged and is **correct**: `ImageNetVal` sorts its paths, `cka_similarity/workers/common.py:31-37` returns `range(25000)` over that order, `s1_within_arch_invariance.py:123-135` walks it (and says in a comment that it is *not* `get_dataset`'s seeded `random_split`), and `s2_cross_architecture.py` slices it from 0; `cross_model_experiment.py` — the loader that does use a stratified/`randperm` subset — is not the driver of these studies. The clause is restored at `study4_cross_arch.tex` and `empirical_setup.tex` now states the whole rule: Studies 1 and 3 and the Phase-1 set use the sorted-filename prefix; only the 200-pair theorem45 set is a seeded `random.Random(42)` draw from a split half. Related correction: `study1_invariance.tex` no longer says the 25,000-image set is "fixed across all experiments in this paper" (false for Study 2).
- **`empirical_setup.tex` corrected on four counts** (the guard `d_f > 1e-6·max d_f` is a numerical guard, not an attack-success filter; the Phase-1 generator overrides **five** attacks — PGD 10→7, CW 50→100, DeepFool 50→100, APGD 10→50 and ce→dlr, Square 5000→20000 — with FGSM alone at torchattacks 3.5.1 defaults, while AlexNet/VGG ran the library defaults; per-attack JSONs carry no hyperparameters; the sampling rule above). The override clause is now **derived** by the script from the generator's attack map and an enforced check fails the run if the emitted caption omits any of it.
- **Series edits I.1–I.10** (Tasks C1–C4 + fix1): the series statement 0/I/II, forward pointers to HAaNE II, the A1–A6 experiment table (Table 1, now referenced from "Thesis and study map"), the TMLR-shaped opening, the section relabelling A1–A6, the breakable claims box, HAaNE expanded at first use.
- **The vocabulary appendix (C5)** is in: **Appendix D, "Vocabulary for algebraists", pp. 40–45**. `sections/vocab/` is a **byte-identical mirror** of the shared source `docs/vocab/` in the Neural-Networks-Matrices repository — `vocab-macros.tex`, the selection file `appendix_vocab_I.tex`, and the **16** entries that file inputs. Re-sync is a plain `cp`; never edit the copy. Only those 16 entries are mirrored: 14 of the 31 entries HAaNE I does not use carry vocabulary reserved to HAaNE II, which this paper must not ship. **Caveat:** two of the 16 entries carry statements the C-s3 fix rounds removed from the body (the attack filtering convention, and "fixed for the whole paper" about the sample set) — they must be fixed in the shared source before submission; see checklist item 1, which is a factual fix, not a cosmetic one.
- **Deferred layout/consistency items (C6):** Figure 1 no longer splits the claims box (`[t]` → `[!ht]`; the box now runs whole and the figure follows it); the Table 1 caption no longer sits on its `\toprule`; the S1 table float is `[htbp]` instead of `[h]`; the Remark in Section 2 is retitled "The quiver lift is scaffolding for future work…" (the retired "forward-series" wording).
- **The June cut list (old LEFT item 4) is applied** — verified 2026-09-05: `study1_invariance.tex` is 150 lines and the two flagged passages are gone; the "130× / identical to numerical precision" framing is replaced (see the header of `study1ab_snippet.tex`); `mathematical_background.tex` no longer exists; the wide-face permutation is described correctly at `study1ab_snippet.tex:31`.

**LEFT (ordered):**

1. **Close the three open review flags** — each is a decision or a run, not writing:
   - **(2) cross-dimensional similarity panel** (Section 7): CKA / Bures / distance correlation are computed on unequal dimensions with no PCA; the methodology, especially Bures, must be confirmed.
   - **(3) missing controls** (Section 7): the Cui/Murphy controls were not run for the cross-architecture study.
   - **(5) unverified figure** (Figure `fig:lp-pathology`, Section 8): the LP-counterfactual figure predates this revision; its caption numbers must be checked against the run and the figure regenerated if needed.
   - **(4) is partly closed:** the appendix ordering panel now has CIs. Still open — the coherence medians of `tab:coherence` (the theorem45 reduce stored aggregates only, so CIs need the pairs re-run, not a file resampled) and the mechanism partial correlations, whose per-pair data **are** on disk (`experiments/vgg_imagenet/theorem45/debug_gamma_zero_*.json` and siblings).
2. **Optional local pulls** — neither is on disk as of 2026-09-05: `results/cross_model/` is empty (Step E per-pair JSONs) and `results/teleportation/` holds only the `*_N500_pre-rerun.json` trio (the N=1000 rerun). Neither blocks the build: "Step E" appears **nowhere** in the paper source, so it is already effectively dropped, and the teleportation trio numbers in the text are the ones the paper currently cites. Adopting the N=1000 rerun would be a data change to Study 1b — Marco's call.
3. **Cover letter** → a **continuation** letter (0 is a separate paper), not a rebuttal ledger; TMLR prior-rejection disclosure = Marco's call.
4. **Page budget.** 48 pp, of which 30 are main matter and 18 are appendices + references. The June target was ≤ 12 pp main; TMLR has no hard limit, but if the appendices are to be trimmed, Appendix D (6 pp) is the newest and the most compressible — the entry set is a per-paper selection, not a fixed list.
5. **Pre-submission cleanup** (checklist below). Item 1 there is **not** cosmetic: two shared-vocabulary entries contradict the corrected body text and have to be fixed in the shared source. The rest is the review apparatus coming out.
6. **Final build, then submit.**

**BLOCKED ON:** nothing. All cluster work is complete (06-23); every remaining item is a local decision, a local run, or the commit.

---

## Pre-submission checklist (do these last, in this order)

1. **Fix two statements the vocabulary appendix imports that contradict the corrected body text.** Both live in the *shared* source `docs/vocab/entries/` in the Neural-Networks-Matrices repository, so they must be fixed **there** and the mirror re-copied — never patched in `sections/vocab/`, which would fork the shared source and be silently clobbered by the next `cp`. Found 2026-09-05 while wiring C5; not fixable inside Plan C's Task C5 scope, which owns the mirror but not the source.
   - `entries/adversarial-pair-attack-family.tex`, *Computed here*: it names only "attack-success filter $d_f\ge1$, $d_M>0$" and its *Validated here* field points that at "Study 2 (coherence and attack-family ordering)". The paper uses **two** conventions and says so explicitly: the headline coherence and ordering tables apply the per-cell relative **division guard** $d_f > 10^{-6}\max_i d_{f,i}$, which `study2_coherence.tex:25-27` calls "*not* an attack-success criterion"; the stricter $d_f\ge1$, $d_M>0$ filter belongs to the appendix Phase-1 panel (`tables/s3_table.tex`) and the mechanism pilot. Suggested replacement clause: "…(six architectures in all for the concordance). Two filtering conventions are used and must not be conflated: the headline coherence and ordering tables apply a per-cell relative division guard $d_f > 10^{-6}\max_i d_{f,i}$, which discards only pairs whose logits did not move; the appendix Phase-1 ordering panel and the mechanism pilot apply the stricter attack-success filter $d_f\ge1$, $d_M>0$."
   - `entries/held-out-validation.tex`, *Definition*: "HAaNE I evaluates on the ImageNet validation images (the first $N$ by sorted filename, **fixed for the whole paper**)". The last clause is exactly the sentence deviation D2-1 removed from `study1_invariance.tex` in C-s3 fix round 2, and `empirical_setup.tex:115-123` now says the opposite: the sample sets "are not all the same set" — Studies 1 and 3 and the Phase-1 pair set use the sorted-filename prefix, while the 200-pair `theorem45` set is a seeded `random.Random(42)` draw from a 25,000-image half. Suggested replacement clause: "…(the first $N$ by sorted filename for Studies 1 and 3 and for the Phase-1 pair set; the 200-pair theorem45 set is a seeded draw from a split half)."
   Everything else in the 16 mirrored entries was checked against the body on 2026-09-05 and agrees — including the float32 row-sum residual ("$\le10^{-2}$ relative to the logit norm, $\sim10^{-10}$ at float64", matching `empirical_setup.tex:105-108`) and the 54 linear programs of the counterfactual.
2. Remove `\input{sections/review_flags}` from `main.tex` (line 21, with its comment line above it) and delete `sections/review_flags.tex`.
3. Remove the three remaining in-text `\WARNING{...}` blocks — `study2_coherence.tex:145`, `study3_visualization.tex:140`, `study4_cross_arch.tex:93` — after their flags (2), (3), (5) are closed, and then the `\WARNING` macro definition in `preamble.tex:47`.
4. Confirm `\TODO` renders nowhere (`grep -rn '\\TODO' sections/ tables/` — clean on 2026-09-05).
5. Decide the five table files that no source `\input`s — `isomorphism_summary.tex`, `teleportation_summary.tex`, `theorem45_{densenet121,googlenet,resnet152}_imagenet.tex`, `theorem45_summary.tex`. They are staged by the `tables/*.tex` glob of the commit recipe; keep them as provenance or delete them, but do not leave the question open at submission.
6. Two floats that reach the PDF are never `\ref`-ed: **Figure 1** (`fig:germ`) and the appendix table **`tab:coherence-max`** (`\input` from `main.tex:43`). Either reference them from the text or decide they stand alone. (Every other unreferenced label lives in one of the five dead table files of item 5.)
7. Optional: add the question sentence and a series line to `abstract.tex`; the series paragraph currently lives only in the introduction.
8. Rebuild and re-run the gates (0 errors / 0 undefined / 0 multiply-defined; page count; overfull ≤ 23; the series-vocabulary greps at the top of this board).

---

## The question and the claims (TMLR structure), with readiness

**Question.** At one trained network, what is the knowledge matrix `M(x)` — what determines it, what it is invariant to, what it determines, and what its geometry measures — and what does that buy that hidden activations cannot provide?

| Claim | Support | Points at | State |
|---|---|---|---|
| Germ identity `M(x) = [Df·diag(x) | f − Df·x]` on the regular set | theorem | `section_germ.tex` (Germ identity) | ✅ proved + verified |
| Maximal invariance: `M` is invariant under the entire stabiliser of the realised germ; permutation invariance at every input | theorem + proposition | `section_germ.tex` | ✅ |
| Completeness: activations are gauge-covariant and germ-incomplete; `(D(x), M(x))` is complete | theorem | `section_germ.tex` ("Hidden activations are not enough") | ✅ |
| KM = gradient×input ⊕ bias attribution; computable by C VJPs, 50–150× cheaper at ImageNet (296× measured) | theorem + timing demo D1 | `section_germ.tex`; `weff_vjp_demo_results.json` | ✅ |
| The per-input contraction is lossy (non-isomorphic nets can share the whole KM field) | proposition | `section_germ.tex` | ✅ |
| Visible/invisible (Pythagorean) decomposition; coherence `A = (d_f/d_M)²` with reference lines `A = 1`, `A ≤ d` | theorem + corollaries | `section_geometry.tex` | ✅ |
| Within-region anatomy; smooth + crossing anatomy along an input path | theorems | `section_geometry.tex` | ✅ |
| No C⁰ bound; three-tier teleportation claim structure | theorem + proposition | `section_geometry.tex` | ✅ |
| Penultimate distances carry no germ-level accounting | proposition | `section_geometry.tex` | ✅ |
| **A1** permutations move the KM by 0.1–0.2% of its adversarial signal; penultimate by ~100% | experiment | Study 1a tables | ✅ local data |
| **A2** teleportation under the three-tier claim structure: exact invariance (theorem), visible drift = logit gate (equality, measured ≤ 1e-1), invisible part unbounded and left unestimated | theorem + experiment | Study 1b | ✅ PATH B framing (06-23); D2 measured-drift column dropped by decision |
| **A3** nine-measure panel with Cui/Murphy controls behaves | experiment | s1/s2 tables | ✅ reduce green, controls pass |
| **A4** attack-family ordering stable across architectures (six-rater Kendall W = 0.921); appendix cross-check on the population-matched Phase-1 trio W = 0.97 (p = 3.1e-05) with bootstrap CIs; median coherence A ≤ 0.23 | experiment | Study 2 tables + `tables/s3_table.tex` | ✅ regenerated 2026-09-05; ResNet-152's top four unresolved by the bootstrap and printed as such |
| **A5** alignment-free cross-architecture comparison on the pretrained trio, side by side with the panel; "not a quality ranking" | experiment | Study 3 (`study4_cross_arch.tex`), s2 table | ✅ data; review flags (2)(3) still open on this section |
| **A6** honest negatives: bake-off (penultimate wins 5/6), matrix-direction counterfactual 0/54, approximate teleportation | experiment | `study3_visualization.tex` | ✅ (figure caption unverified — flag (5)) |
| **Not claimed:** superiority over penultimate; uniqueness of the invariance; row sums exact in finite arithmetic; zero teleportation drift; anything about populations of training runs (HAaNE II's) | — | intro not-claimed list | rule |

---

## Sections (files under `sections/`)

| File | Content | State | Next edit |
|---|---|---|---|
| `abstract.tex` | abstract | drafted | optional: question sentence + series line |
| `introduction.tex` | intro + series statement + Table 1 (A1–A6) + claims box + not-claimed | ✅ series edits applied | — |
| `fig1_germ_identity.tex` | Fig. 1 (`[!ht]`, follows the claims box) | ✅ | never `\ref`-ed — see checklist item 5 |
| `section_germ.tex` | Thms germ / max-inv / completeness / W_eff / lossy | ✅ | — |
| `section_geometry.tex` | Pythagoras / anatomy / no-C⁰ / teleportation tiers | ✅ | — |
| `empirical_setup.tex` | setup, float32 residual table, sampling rule, attack budgets | ✅ corrected 2026-09-05 | — |
| `study1_invariance.tex`, `study1ab_snippet.tex` | A1 + A2 | ✅ PATH B reframing; cut list applied | — |
| `study2_coherence.tex` | A4 | ✅ ordering narrative regenerated | flag (4): coherence-median CIs |
| `study4_cross_arch.tex` | A5 (titled Study 3 in text) | drafted | flags (2) and (3) |
| `study3_visualization.tex` | A6 honest negatives | drafted | flag (5): LP figure |
| `limitations.tex`, `conclusion.tex` | — | ✅ forward pointers to HAaNE II | — |
| `appendix_proofs.tex`, `appendix_A_geometric.tex`, `appendix_engineering.tex` | proofs, geometry, engineering | ✅ | — |
| **`vocab/appendix_vocab_I.tex` + `vocab/vocab-macros.tex` + `vocab/entries/` (16)** | Appendix D — vocabulary for algebraists | ✅ **added 2026-09-05**; byte-identical mirror of `docs/vocab/` in the NNM repo | re-sync by `cp`; never edit the copy |
| `review_flags.tex` | internal flags | review-only, still `\input` at `main.tex:21` | **remove before submission** |

**How the vocabulary appendix is wired** (so it stays droppable by hand): `main.tex` sets
`\providecommand{\vocabdir}{sections/vocab}` immediately before `\input{sections/vocab/appendix_vocab_I}`,
after `appendix_A_geometric`. LaTeX resolves `\input` against the pdflatex working directory, not the
including file, so `\vocabdir` — and nothing inside the mirror — carries the host path; the selection file's
own default (`../vocab`) is written for a sibling-directory mirror and is left untouched. The selection file
inputs `vocab-macros.tex` itself and re-binds `\KM`, `\vx`, `\vone` to this paper's notation inside a
`\begingroup … \endgroup`; **do not** `\input{vocab-macros}` from `preamble.tex` — the re-binding has to stay
scoped to the appendix. Collision check against this preamble (2026-09-05): `\R` is defined identically
(`\mathbb{R}`) so the `\providecommand` is a no-op; `\KM` is defined here as `\textsc{KM}` and is **never used
in the body**, so the group's `\renewcommand{\KM}{\mathrm{M}}` is invisible outside the appendix; `\vx` and
`\vone` are not defined here at all; `\vocabentry`, `\vocabsee`, `\vocabdir`, `\vocablabel` are not defined
here; no other `app:vocab` label exists; and no entry uses this paper's `\M`, `\Mfx`, `\Weff`, `\beff`,
`\TODO` or `\WARNING`. Verified in the PDF: the entries render as `M(x)`, plain `x`, `\mathbf 1`.

## Data and artefacts

| Item | Where | State |
|---|---|---|
| Theorem-4.5 / coherence JSONs, 7 settings | `experiments/*/`, `cluster-data/` | local |
| Phase-1 aggregated panel + sanity report | `cluster-data/results/phase1/aggregated/`, `cluster-data/sanity_report.json` | local (06-19 tarball) |
| Ordering-table regeneration script | `scripts/regen_ordering_tables.py` | local; deterministic (`--date`, `--seed`), 80 enforced self-checks; re-emits `tables/s3_table.tex` byte-identically |
| Teleportation JSONs | `results/teleportation/` — N=500 pre-rerun trio + resnet18/resnet50 | local; the N=1000 rerun is **not** on disk |
| D2 KM-drift (densenet121, resnet152) | `cluster-data/corrupted/*_km_drift.json` (name says corrupted — *verify* which copy is the good one) | local; column dropped from the paper by PATH B |
| Step E per-pair JSONs | Nibi `results/cross_model/{arch}/per_pair/` | **not pulled**; `results/cross_model/` is empty locally, and "Step E" appears nowhere in the paper |
| VJP demo | `../../../resubmission-artifacts-2026-06-11/weff_vjp_demo_results.json` | local |
| Plan-C workspace (briefs, reports, snapshots, patches, commit recipe) | `…/Neural-Networks-Matrices/.claude/worktrees/learning-mechanics/.superpowers/sdd/2026-09-04-plan-C-haane-i-series-edits/` | local (other repo) |

## Provenance rules (from `CLAUDE.md`, unchanged)

No raw/RMS amplification headlines; no "superior to penultimate"; no theorem-asserted zeros; no "identical to numerical precision"; permutation invariance cited to the new theorems, not Thms 4.1/4.2 of HAaNE 0; every number in the abstract has a local artefact behind it.
