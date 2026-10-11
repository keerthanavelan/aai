# Attack-aware revision: one command

1. Let any existing experiment finish. Stop app.py with Ctrl+C before replacing
   code. Keep data/, results/, models/ and venv/ untouched.
2. Extract this ZIP. Copy its src folder, app.py and run_improved.py into your
   existing aai folder, replacing code files when Windows asks. Copy the updated
   documentation too. Do not create an extra nested project folder.
3. Use a terminal whose path ends in aai. Run:

       .\venv\Scripts\python.exe run_improved.py

4. Wait for FINAL TEST RESULTS. The exact combined report filename is printed.
5. Assess full vs GA/PSO accuracy, attack recall, false alarms and feature counts.
   Do not select the highest test-scoring seed; select by validation fitness.

## What changed

- Original random-split accuracy scoring remains available through objective=accuracy.
- objective=attack_aware emphasizes mean recall across validation attack types,
  overall attack recall and normal specificity. The score is
  0.5*attack_macro_recall + 0.25*attack_recall + 0.25*(1-false_alarm_rate).
- Revised fitness = 0.98*(1-security_score) + 0.02*(feature_count/41), with the
  existing feasibility penalties. This replaces the 10% feature penalty in the
  revised preset; it does not silently relabel the original experiment.
- At least eight original inputs are retained in the revised preset. This is
  an explicit chosen search constraint, not a discovered minimum.
- The search sample includes available rare attack labels, using training data
  only. Validation sampling includes all rows in this preset.
- RF attack-type sample weighting improves the representation of rare attacks.
  attack_type_sqrt uses inverse-square-root type frequency; attack_type_equal
  uses inverse type frequency. Weights are normalized within attacks to preserve
  total normal-vs-attack weight. Attack types are not classifier input columns.
- max_depth=null allows full-depth trees. rf_max_features=0.5 examines more
  encoded inputs at each split. All methods use the same chosen RF settings.
- Constant original inputs are excluded from GA/PSO/chi-square search. The full
  baseline still uses all 41 original inputs for the stated comparison.
- Both optimizers begin with a mix of RF-ranked masks and random positions.
  Ranked initialization is an added filter/wrapper combination and must be
  disclosed in the method. Their timing includes this initialization.
- src/calibration.py selects weighting mode and threshold using the fixed
  KDDTrain+ training/validation split, with a 0.5% validation false-alarm ceiling.
  The selected configuration and all calibration candidates are saved. No test
  labels/scores are used by this selection procedure.
- The chosen configuration is frozen before the final test evaluation.
- Mean recall across attack types gives rare classes equal influence but can
  be noisy for very small support. Always show per-type sample counts.

## Important interpretation

The original test results showed low attack recall even with all inputs. More
GA generations alone cannot repair that. This revision targets the scoring and
classifier behavior; it does not promise 95%/99% test accuracy.

The same official test set has already been inspected across revisions. Treat
the revised results as exploratory benchmark comparisons, not a newly untouched
test estimate. A stronger research study would preregister model selection or
use another independent dataset for confirmation. Do not combine original and
revised runs into one mean/std or significance calculation.

Modern-network coverage and deployment robustness are not established by this
historical dataset. A threshold chosen on validation may produce a higher
false-alarm rate on the test distribution; report that tradeoff openly.

Person 2's existing API contract remains usable. New configurable fields are
objective, weight_mode, rf_max_features, exclude_constant and seed_ranked;
max_depth now also accepts null. Snapshot/validation metrics add
attack_macro_recall. Existing saved models still load.
