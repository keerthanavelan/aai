# Use existing models without training again

Run in the existing project folder after pulling this update:

```powershell
.\venv\Scripts\python.exe ensemble_saved.py --report results/experiment_31e9b36582044da995b33f1b20eb6c92.json
```

This loads all GA models from that experiment, averages their attack scores
equally and applies the existing experiment threshold. It never fits a model,
repeats a search or tunes weights/thresholds using test labels. It reads existing
prediction caches, checks live predictions match those caches, saves one extra
ensemble model and prints accuracy, attack recall, F1 and false-alarm rate.
Runtime depends on model loading, prediction and compression, rather than search.
Accuracy can decrease; crossing 90% is not guaranteed.

This is exploratory: the idea was introduced after inspecting the test results.
Keep the original five-seed comparison and this extra ensemble result separate.
Do not replace the original experiment statistics with an ensemble score.

Restart Flask after saving. `/api/models` lists the new `ga_ensemble` entry.
Use its returned model_id in the existing classify, CSV, replay, what-if and
analyst-challenge routes. No API redesign is required. It needs all input
features in the union of the member masks; report that count, not the average
member feature count. It averages model scores, not calibrated confidence.
More forests can increase inference time; actual measured batch timing is saved.

Frontend notes: `validation_fitness` is null because the ensemble has not been
scored by the optimizer. Display "Not applicable". Do not invent an ensemble
convergence curve or equate its security constraints with its members'.

Verified with the existing small two-seed integration models: cached score
agreement, classifier entry point, model serialization/reload and model listing.
That verifies functionality, not the latest five-seed ensemble's performance.
