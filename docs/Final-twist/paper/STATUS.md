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

**Last updated:** 2026-09-07 — **Teleportation is EXACT; the paper's "function-approximate" claim is
withdrawn.** See the dedicated section at the bottom of this board. Net effect: Study 1b becomes a genuine
multi-architecture invariance verification, Limitation **L1 is withdrawn**, and the third honest negative
(approximate teleportation) is **withdrawn**. New `sections/appendix_teleport_exact.tex` (+ two repro
scripts) carries the evidence. Previous update: 2026-09-05 — Plan C is **complete**: series edits C1–C4 (+ fix round 1), the ordering-table
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
| **A2** teleportation is an **exact** isomorphism (BN in eval mode included); KM drift measured at the fp64 floor, 0.8–2.5e-15 relative, on all three archs | theorem + experiment | Study 1b, `appendix_teleport_exact.tex` | ✅ **corrected 2026-09-07** (was: "three-tier, visible drift = logit gate, invisible unestimated") |
| **A3** nine-measure panel with Cui/Murphy controls behaves | experiment | s1/s2 tables | ✅ reduce green, controls pass |
| **A4** attack-family ordering stable across architectures (six-rater Kendall W = 0.921); appendix cross-check on the population-matched Phase-1 trio W = 0.97 (p = 3.1e-05) with bootstrap CIs; median coherence A ≤ 0.23 | experiment | Study 2 tables + `tables/s3_table.tex` | ✅ regenerated 2026-09-05; ResNet-152's top four unresolved by the bootstrap and printed as such |
| **A5** alignment-free cross-architecture comparison on the pretrained trio, side by side with the panel; "not a quality ranking" | experiment | Study 3 (`study4_cross_arch.tex`), s2 table | ✅ data; review flags (2)(3) still open on this section |
| **A6** honest negatives: bake-off (penultimate wins 5/6), matrix-direction counterfactual 0/54, plus one methodological commitment (drift is measured, never asserted from a theorem) | experiment | `study3_visualization.tex` | ✅ (figure caption unverified — flag (5)); third negative (approximate teleportation) **withdrawn and rewritten** 2026-09-07 |
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


---

## Teleportation exactness — correction of 2026-09-07

**What was wrong.** The paper claimed neural teleportation on these BatchNorm networks is
*function-approximate*, and hung a three-tier claim structure, Limitation L1, and one of the three honest
negatives on it. The supporting diagnosis (`km-notes.md`, 2026-05-01) was a **conjecture that was never
tested**: *"Most likely cause: BN running-stats are not migrated alongside the COB matrices."*

**What is true.** Teleportation is an **exact** function-preserving isomorphism, BatchNorm in evaluation
mode included. `neuralteleportation` does not migrate the running statistics because it does not need to:
`layers/neuron.py:BatchNormMixin._forward` computes `base_BN(input / prev_cob)`, restoring the original
pre-normalisation activation, then scales `weight`/`bias` by `next_cob`. Composite = `tau_out * BN(z)`,
exactly. (This is *better* than stat migration, which would also need `eps` rescaled.)

**The evidence** (`scripts/verify_teleportation_exactness.py`, `..._km_exactness.py` — CPU, minutes, no
cluster). Same COB draw at fp32 and fp64, with three library `.float()` downcasts patched out first
(`COBForwardMixin.forward`, `merge.py:Add.forward`, the fp32 trace input in
`NeuralTeleportationModel.__init__` — without these the fp64 arm keeps an fp32 floor and the ratio
saturates near 10):

| Arch | fp32 drift | fp64 drift (relative) | fp32/fp64 ratio |
|---|---|---|---|
| resnet152 | 4.2–6.1e-6 | 8.2e-15–1.0e-14 (1.4–1.7e-15) | 4.2–6.7e8 |
| densenet121 | 6.2–10.5e-6 | 1.2–2.3e-14 (1.6–3.2e-15) | 3.3–7.9e8 |
| googlenet | 5.7–8.6e-6 | 1.4–1.7e-14 (1.9–2.2e-15) | 3.4–5.7e8 |

Pure-roundoff prediction `eps32/eps64 = 2^29 = 5.37e8`; a real function change predicts ~1. **KM itself:**
per-class VJP rows at both precisions give fp64 relative drift **0.8–2.5e-15** on all three archs.

**Why the cluster saw 1e-2.** `job_teleportation.sh` requests `--gpus=h100:1`; PyTorch defaults
`torch.backends.cudnn.allow_tf32=True` and this repo never disables it. TF32 `eps = 2^-11 = 4.9e-4`, 8192x
coarser than fp32 — scaling the CPU fp32 drift by 8192 predicts 3.5–5.0e-2 on resnet152 vs **3.1e-2–1.0e-1
observed**. Secondary amplifier: at `cob_range=1` the library samples tau on **[0,2]**, so the smallest of
~2e4 draws is ~3e-5 and the multiply/divide round trip passes through ~1e4.

**Relation to the June PATH B decision.** That diagnostic (`14624489`, 06-23) tested the
`load_cob_into_km` **load path** and correctly found a load bug there (the loaded teleported KM violates
`M·1 = f` structurally). It is not refuted — it is a different code path. The 2026-09-07 test bypasses
`load_cob_into_km` entirely, computing the KM by autograd VJPs on the teleported model. **Caveat on that
test:** it derives the bias column as `f - Jx`, so the row-sum identity holds by construction and is not
independently tested; what it does test — and what matters — is that the Jacobian block `J·diag(x)` is
identical between the original and teleported networks to machine epsilon.

**Prop 2.2(ii) attributed, not claimed (2026-09-08, ninth pass).** Marco asked whether it is known. It is.

- **(ii) is a restatement of a classical fact.** $f(z)/z$ locally constant on $\{z>0\}$ and $\{z<0\}$
  \emph{is} positive homogeneity of degree one, $f(\lambda z)=\lambda f(z)$ for $\lambda>0$; the
  classification of continuous such functions on $\R$ as $a_+\max(z,0)+a_-\min(z,0)$ is elementary and
  long-standing. Searched and verified: the $\R^n$ theory is surveyed by Gorokhovik, *Positively
  Homogeneous Functions Revisited*, JOTA 171(2):481--503, 2016; the activation-level consequence (ReLU and
  leaky ReLU are the positively homogeneous ones) is used routinely --- Neyshabur et al. Path-SGD 2015,
  Dinh et al. 2017, and most explicitly **Lyu \& Li, ICLR 2020**, whose whole framework assumes exactly
  this class.
- **Proposition retitled** "…; (ii) is classical", the equivalence with positive homogeneity stated inline,
  and a new attribution box added. Two bib entries added (`lyu2020gradient`, `gorokhovik2016positively`).
- **What is left as ours, and hedged**: the reformulation as a condition on the slope diagonal $D$, and
  part (iii) --- that the class is strictly smaller than PL, so a PL network can have a germ everywhere and
  still fail the germ identity. Both are worded as "we have not found this stated elsewhere", not as
  results. This keeps the claims-discipline line: no repeat of the retracted "no other practically
  computable representation has this property".

**$f(0)=0$ dropped; the real condition is $X_{\mathrm{nz}}$ (2026-09-08, eighth pass).** Marco: sigmoid
does not satisfy $f(0)=0$ and the condition should go. Correct --- it was never needed for the definition.

- **The KM is now defined for *every* activation, with no condition on $f$.** The hypothesis is
  $x\in X_{\mathrm{nz}}$: no hidden pre-activation vanishes. Verified: on a sigmoid net ($f(0)=\tfrac12$)
  the row sums reproduce the logits to $10^{-14}$ at random inputs.
- **The guard cannot be fixed, and this is not a defect of the guard.** Where $z_q=0$ the post-activation
  is $f(0)$ while $D_{qq}z_q=0$ for *any* finite $D_{qq}$; the shortfall is exactly $f(0)$ whether the
  guard is $0/0\mapsto0$, $\mapsto1$ or $\mapsto10^6$ (all three checked). So $f(0)=0$ is precisely the
  condition under which the exceptional set is empty --- a bonus, not a prerequisite. Reworded that way
  throughout.
- **$X_{\mathrm{nz}}$ is open, dense, full measure** whenever no unit's pre-activation vanishes
  identically: automatic for real-analytic $f$ (sigmoid, tanh, GELU, SiLU); for the ReLU family it is the
  complement of Lemma A.1's hyperplanes, so it costs nothing there.
- **Honest caveat now stated:** when $f(0)\neq0$ the quotient behaves like $f(0)/z$ near the excluded set,
  so $D$ is unbounded there ($5.25$ at $z=10^{-1}$, $5\times10^{5}$ at $z=10^{-6}$ for sigmoid). The
  identity stays exact; the entries do not stay small.
- **Prop 2.2(ii) no longer assumes $f(0)=0$** --- under (LCS) plus continuity it *follows*
  ($f(0)=\lim_{z\to0^+}a_+z=0$), so under (LCS) the $X_{\mathrm{nz}}$ restriction is vacuous and
  Eq. (2) holds everywhere.
- Touched: the setup and both clarify boxes in `section_germ.tex`, the secant box, and the Lemma A.1 scope
  box in `appendix_proofs.tex`.

**(LCS) replaces "PL" as the hypothesis (2026-09-07, seventh pass).** Marco's observation: what the germ
results need is not piecewise linearity but that $D$ depends *discretely* on $x$. He is right, and the
distinction has teeth.

- **New Definition 2.1 (LCS)** --- "locally constant slope diagonal": every $D^{(\ell)}$ of Eq. (1) is
  constant on a neighbourhood of $x$. **New Proposition 2.2** proves (i) (LCS) $\Rightarrow$ $\Psi$ locally
  affine; (ii) for continuous $f$ with $f(0)=0$, (LCS) holds off a null set **iff**
  $f(z)=a_+\max(z,0)+a_-\min(z,0)$ --- the two-parameter leaky-ReLU family (each linear piece must pass
  through the origin, and continuity then forces the only breakpoint to be $z=0$); (iii) (LCS) is
  **strictly stronger than PL**.
- **Why it has teeth.** hard-tanh $\mathrm{clip}(z,-1,1)$ and $\max(z-1,0)$ are piecewise linear, so their
  networks are piecewise affine and have germs, **yet Theorem 2.4 fails for them**: at $z=2$ hard-tanh has
  $f(z)/z=1/2$ against $f'(z)=0$. Verified numerically end-to-end ($|J_{\rm sec}-D\Psi|=1.67$ on a small
  hard-tanh net, $0$ for ReLU / LeakyReLU / abs; the row-sum identity holds at $10^{-16}$ for all of them).
  So the old "PL networks" hypothesis was **too weak** for Theorems 2.4/2.7/2.10, though it gave the right
  answer for our architectures, which are all ReLU.
- **Part (ii) explains Lemma A.1's structure**: under (LCS) an activation can break only at $z=0$, which is
  why the walls are the zero sets $\{z^{(\ell)}_{\sigma,q}=0\}$ and nothing else.
- **Hypotheses restated on (LCS)**: Thm 2.4 (germ identity), Thm 2.7 (maximal invariance --- was "any two
  PL networks"), Thm 2.10 (gradient$\times$input equivalence), Lemma A.1, the abstract, the introduction's
  claim list, and three places in `appendix_engineering.tex`. "PL" is kept as descriptive shorthand for our
  setting (which does satisfy (LCS)) but no longer carries a hypothesis anywhere.
- **Theorem numbers shifted** by the two new statements: germ identity 2.2$\to$2.4, maximal invariance
  2.5$\to$2.7, completeness 2.7$\to$2.9, gradient$\times$input 2.8$\to$2.10. All refs are `\ref`-based and
  resolve; no hardcoded numbers in the source.

**The knowledge matrix is defined for ANY activation (2026-09-07, sixth pass).** Marco's correction, and
it is a real improvement to the logical order.

- **$D^{(\ell)}(x)$ is now defined generally**, as the diagonal of activation-to-pre-activation quotients
  $f(z_q)/z_q$ (guard $0/0\mapsto0$), for any activation with $f(0)=0$ --- new Eq. (1) in
  `section_germ.tex`. The $0/1$ mask is the ReLU **specialisation**, not the definition. Eq. (2)
  (`eq:km-def`) is therefore the definition of the KM for an arbitrary network, and its row-sum identity
  carries no genericity hypothesis.
- **Backed by Armenta--Jodoin.** This is exactly their *induced thin quiver representation* $W^f_x$, and
  their **Theorem 6.4** states $\Psi(W^f_x,\mathbf 1)(\mathbf 1_d)=\Psi(W,f)(x)$ for any activation. Now
  cited at the definition. (Their zero-pre-activation guard adds $\eta\ne0$ rather than $0/0\mapsto0$;
  the two agree whenever $f(0)=0$, which both require.)
- **Theorem 2.2 retitled "Germ identity: the piecewise-linear case"** with the PL hypothesis stated in the
  theorem rather than assumed by the surrounding prose. The germ reading is a *theorem about the PL case*,
  not part of the definition.
- **Scope box added after Lemma A.1** answering the question directly: the lemma does **not** survive the
  general definition, and not marginally --- for a strictly nonlinear smooth $f$ the pattern is locally
  constant essentially nowhere, so $X_{\mathrm{reg}}$ is generically **empty** rather than full-measure,
  and there is no switching set to bound. Includes the verified counterexample: at $x_0=1$ a single
  $\tanh$ unit and the affine map with the same germ $(0.419974,\,0.761594)$ give KMs
  $[\,0.761594\mid0\,]$ vs $[\,0.419974\mid0.341620\,]$ --- same germ, same row sum, different matrix;
  the ReLU control returns $[\,1\mid0\,]$ for both. So Theorems 2.2/2.5/2.7 are PL statements. Whether
  weaker function-determination survives for non-PL is flagged **open**, not claimed.
- The secant clarifybox after Lemma 2.4 no longer calls itself an "extension" of Eq. (2) --- it *is* its
  content; the masked-product reading is the specialisation.

**Notation rename + vocabulary fork (2026-09-07, fifth pass).**

- **`d_f` → `d_\Psi`** throughout: 70 replacements across 10 section files and 9 table files. LaTeX
  comments describing the code were left alone (they name the on-disk JSON keys). **The generators were
  updated too**, so the rename survives regeneration: `scripts/regen_ordering_tables.py` (8 LaTeX-emitting
  strings + a header note that the JSON key stays `d_f` while the symbol is `d_\Psi`),
  `generate_theorem45_tables.py` (2), and `../resubmission-artifacts-2026-06-11/scripts/make_study2_tables.py`
  (3). **Verified:** re-running `regen_ordering_tables.py --out-dir <tmp> --date 2026-09-07` passes all its
  self-checks and its `s3_table.tex` is byte-identical to the committed one apart from the date stamp.
- **`f_\theta` → `\Psi(W\!,f)` in the vocabulary appendix**, on Marco's instruction ("θ is the parameters
  and W is the weights so θ = W"). Nine entries touched. Also `\KM\vone=f` → `\KM\vone=\Psi(W\!,f)` in the
  row-sum entry's title and its two cross-references, plus `adversarial-pair-attack-family` (which
  spells out HAaNE I's own coherence notation: `d_f`→`d_\Psi`, the division guard, and the
  attack-success filter), `e^{f_k}`→`e^{\Psi_k}`, `p(f+\kappa\vone_C)`→
  `p(\Psi+\kappa\vone_C)`, `\mathrm{Stab}(f)`→`\mathrm{Stab}(\Psi)`, `\{\theta\}`→`\{W\}`.
- **⚠️ The vocab mirror is now a FORK.** It was byte-equal to `docs/vocab/` in the HAaNE II repo; it is not
  any more. A naive `cp` re-sync reverts everything above. Port the notation upstream, or re-apply after
  any sync. Recorded in CLAUDE.md too.

**Limitations renumbered + Kendall's $W$ expanded (2026-09-07, fourth pass).**

- **L4 folded into L1** (Marco's call). Its content survives as L1's second paragraph: the permutation
  check covers ResNet-152 only because a post-pool channel permutation does not compose through
  DenseNet's concatenation or GoogLeNet's Inception branches; teleportation handles both natively; a
  concat-aware permutation would extend a *correctness check*, not add evidence. `closes L4` removed
  from the conclusion.
- **Ls renumbered contiguous.** Was L1, L1b, L2, L3, L5–L9; now **L1–L9**. Only three moved:
  L1b→L2, L2→L3, L3→L4; L5–L9 unchanged. All cross-references updated and verified
  (L1: appendix_teleport_exact, conclusion, study1ab ×2; L4: conclusion ×2, study3_visualization ×2).
- **Kendall's $W$ box rewritten as a worked example** (`study2_coherence.tex`), answering the
  "I still don't understand it" list: cells = (architecture, attack) squares; $m=6$ raters =
  architectures, $n=6$ items = attack families; descending means rank 1 = largest median $d_M/d_f$;
  midranks defined; **the actual $6\times6$ rank matrix is now printed in the box** with its column sums
  $C=(24,34,15,7,32,14)$, $\bar C=21$, $S=580$, $S_{\max}=630$, $T=0$ (no ties, $1^3-1=0$),
  denominator $6^2(6^3-6)=7560$, and $W=S/S_{\max}=580/630=0.9206$ — the intuitive form, shown equal to
  the textbook $12S/(m^2(n^3-n)-mT)$. Monte-Carlo explained as sampling *whole random rank matrices*
  ($B=10^6$, 0 hits); the $m=3$ null enumerated exactly over $(6!)^2=518{,}400$, attaining 77 distinct
  $W$ values with closest pair 0.0127 — hence two decimals. Also states what $W$ ignores: gap sizes
  (1.13–1.51× and 1.57–2.09×) and any monotone transform, which is why the raw-vs-RMS unit choice
  (dividing $d_M/d_f$ by $\sqrt{d+1}$, the same constant across a row) cannot move it.
- **Verified $W=0.921$ reproduces** from `experiments/*/theorem45/theorem45_results.json`: recomputing
  the midranks from the cell medians gives $W=0.920635$ and $T=0$. (An intermediate hand-transcription
  of GoogLeNet's row gave 0.930; the published table and the reduce are both correct — GoogLeNet has
  PGD 2.221 > APGD 2.096.)

**Study 1 restructured (2026-09-07, third pass).** Marco's call: the permutation and teleportation
*measurements* were unit tests and are demoted to the appendix; Study 1 is now **"what the standard
similarity measures do with a gauge transformation"**, i.e. the nine-measure panel.

- **New §5.1 "Why the invariance itself needs no experiment"** — one analytic subsection replacing the two
  measurement studies. Cites **Armenta--Jodoin Thm 4.13** ($\Psi(W,f)=\Psi(V,g)$ under any isomorphism) for
  exactness, **Remark 5.4** (at test time BN's running mean/var are ordinary weights, so BN is inside the
  framework), and **Eq. (2)** $(\tau\cdot f)_v(z)=\tau_v f_v(z/\tau_v)$ — which is *literally* what
  `BatchNormMixin._forward` computes. Penultimate drift given in closed form: permutation reindexes,
  teleportation gives $h\mapsto\tau\odot h$ so drift $=\|(\tau-1)\odot h\|_2/\sqrt D$.
- **Verified numerically that the closed form is exact** (`scratchpad/penult_tau.py`): $h_\tau=\tau\odot h$
  to 1.3–1.7e-6 relative (fp32 eps), and the closed form reproduces the measured RMS drift to 1e-9
  relative. Also drift/RMS(h) = 0.60–0.68 vs the a-priori $1/\sqrt3=0.577$ for $\tau\sim U(0,2)$.
  **Consequence:** the km-notes "natural-vs-random inversion" curvature story is retracted — it is just the
  ordering of penultimate feature norms.
- **Old 1A/1B moved into Appendix C** (`app:teleport-checks`), retitled as implementation checks, with
  `table_signal_relative` following them. Appendix C's opening now *cites* Thm 4.13 rather than deriving
  exactness, and its BN subsection is reframed around Eq. (2) instead of "the library takes a cleaner route".
- **Experiment table A1–A6 → A1–A4** (old A3→A1 panel, A4→A2 coherence, A5→A3 alignment-free,
  A6→A4 negatives); all `sec:A*` labels and claims-box bullets renumbered.
- **Bibliography fixed:** `armenta2021representation` had the wrong arXiv id (2104.14082); it is
  **arXiv:2007.12213**, published *Mathematics* **9**(24):3216, 2021, doi 10.3390/math9243216.
  `armenta2024neural` relabelled as software (`howpublished={Software}`).
- **Added:** $C^0$-closeness definition at Thm 3.6 (it was used 3× and never defined); a full
  **Kendall's $W$** box giving the formula $W=12S/\left(m^2(n^3-n)-mT\right)$, the midrank construction,
  the tie correction, and both null constructions (MC $B=10^6$ for $m=6$; exact $(6!)^2$ enumeration for
  $m=3$). Noted Armenta--Jodoin's max-pooling caveat next to Thm 2.8's tie hypothesis.

**Applied into the main text (2026-09-07, second pass).** The corrections are no longer carried as blue
"this was wrong" notes — the black text now states the corrected claims directly, and the correction
scaffolding is gone. Blue remains only where it is genuinely new or definitional (the author will edit
those by hand). Specifically: Study 1B retitled *"teleportation, an exact multi-architecture isomorphism"*
and rewritten; Study 1's opening framing rewritten; Study 1C's three teleportation paragraphs rewritten;
`introduction.tex` study-map and honest-negatives list; `conclusion.tex`; `section_geometry.tex` (tier (i));
the honest-negatives third item converted from a withdrawn negative into a **methodological commitment**
("Knowledge-matrix drift is measured, never asserted from a theorem") and the section's count changed from
"three places" to "two places … and one methodological commitment"; the `n<p` sentence in Study 1C fixed
(and the duplicate note in `empirical_setup.tex` trimmed to the definition); a step added to Study 1B's
procedure for the fp64 KM measurement. **L1 was repurposed, not deleted:** it is now *"Both invariance arms
are quiver isomorphisms; the cross-architecture claim has no transform-based evidence"* — the real residual
limitation — with a new **L1b** covering the TF32 resolution limit, the CPU-only scope of the fp64
verification (5 draws × 8 inputs, not the 50×25,000 panel), and the unrepaired `load_cob_into_km` defect.

**Paper edits, first pass (2026-09-07).** `appendix_teleport_exact.tex` (new, Appendix C);
`study1ab_snippet.tex` (Study 1B retitled + reframed); `study1_invariance.tex`;
`introduction.tex` (study map, A2 row, both claim-box bullets); `limitations.tex` (**L1 withdrawn**);
`conclusion.tex`; `study3_visualization.tex` (**third negative withdrawn**);
`section_geometry.tex` (which tier applies).

**Cheap follow-up for any rerun:** `torch.backends.cudnn.allow_tf32 = False` (and the matmul flag) and
sample the COB away from zero. **No BN-aware library rewrite is needed — that item is closed.**

### Still open on this
- The `load_cob_into_km` bug itself is unfixed (D2's measured-drift column stays dropped). The paper no
  longer needs it, but the bug is real and should be recorded as such rather than as "teleportation drifts".
- The N=1000 teleportation rerun JSONs are still not on disk (see item 2 above); unaffected by this
  correction, since the corrected claim rests on the precision-scaling test, not on those runs.
