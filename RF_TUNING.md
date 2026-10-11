# Bounded RF tuning of the existing PSO mask

This experiment is completed and its results are in RF_TUNING_EVIDENCE.json.
You do not need to repeat it to finish the project.

Five Random Forest settings were evaluated on the existing training-validation
split, with the frozen 11-feature PSO mask and attack-type weighting unchanged:
150 versus 300 trees; minimum leaf size 1 versus 2; max_features 0.5, 0.75 or sqrt.
Thresholds were fixed in the script before final test evaluation.

The selection rule required validation attack recall >=98%, attack-type macro
recall >=98%, and false-alarm rate <=2%. Among feasible candidates it maximized
F1, then attack recall, then minimized false alarms. Exact ties kept the first
candidate (baseline first). No test labels or test metrics selected the candidate.

The validation choice was 150 trees, sqrt features, leaf size 2, threshold 0.3.
Validation F1 was 99.65% and false-alarm rate was 0.53%. After one refit on all
KDDTrain+, its KDDTest+ accuracy was 84.55%, recall 79.76%, F1 85.46%, and
false-alarm rate 9.10%. The current calibrated detector's measured accuracy
92.47%, recall 94.51%, F1 93.46%, and false-alarm rate 10.23% remain preferable
for this project's recall-focused demonstration. The active detector was not
changed. Better validation performance did not translate to better test performance.

This is an exploratory revision after earlier test-set inspection. The mask and
validation split were previously used, so neither these numbers nor the current
detector's numbers are a fresh independent or nested validation evaluation.
There is no guarantee for arbitrary future input or global-optimum claim.

## Optional reproduction

Only run this if you want a separate local candidate. It does not improve the
completed experiment automatically and it does not change detector_policy.json:

```powershell
.\venv\Scripts\python.exe tune_detector.py --minutes 50
```

The budget includes validation workers and at most one final fit. Workers are
terminated on timeout; disk serialization and process startup can add small overhead.
If no completed candidate improves the validation rule, final fitting is skipped.
New files are confined to results/rf_tuning_<id>/ and models/<new-id>.*.
Original models, results, and the active detector remain unchanged. Partially
written files from an interrupted final fit are not advertised until metadata exists.

Verification: all five full-data validation fits completed; final candidate was
fit and evaluated; a temporary model's save/load and two-connection classification
passed; a hard timeout stopped a worker without advertising a candidate.
