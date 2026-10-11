# Measure operating-point tradeoffs without retraining

```powershell
.\venv\Scripts\python.exe performance_audit.py --report results/experiment_31e9b36582044da995b33f1b20eb6c92.json
```

The script reads saved prediction probabilities, evaluates a fixed grid of
thresholds and writes a CSV for the security-evaluation tab. It prints accuracy,
attack recall, F1 and false-alarm rate at the original threshold and the highest
test-accuracy threshold for each model. It does not fit models or change saved
thresholds. It discovers extra ensemble metadata from the same experiment.

The maximum is explicitly TEST-TUNED: it is useful for exploration, not an
unbiased final test estimate. Do not substitute it for the original experiment
score or claim a new independent above-90% result. Threshold changes can increase
missed attacks while reducing false alarms. Show all metrics together.

The current frontend API already supports threshold overrides and test-threshold
curves. No extra UI architecture is needed; Person 2 can render this tradeoff in
Security Evaluation and let users explore thresholds in Detection.

Local additional investigation: a fixed validation-only comparison of regularized
Random Forest, Extra Trees and histogram boosting on the existing GA validation
winner and the union of GA inputs, with attack-type stress validation, selected
an Extra Trees configuration whose test accuracy was 86.10%. This did not improve
the existing detectors, so that training workflow has not been recommended or
published. The current 90% target remains unverified.
