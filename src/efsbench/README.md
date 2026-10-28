# efsbench

Core routines for evaluating evolutionary feature selection with paired inner and outer scores. Run analyses from the restored project with its recorded environment; the [reproduction instructions](../../REPRODUCE.md) give the collection-specific entry points.

## Modules

| Module | Function |
|---|---|
| `datasets.py` | Download and verify OpenML/UCI sources; load predictors, binary labels, and optional group identifiers. |
| `splits.py` | Construct stratified or grouped folds and derive random seeds from experiment keys. |
| `learners.py` | Fit the fixed ridge and 5-NN classifiers and produce out-of-fold scores. |
| `wrapper.py` | Evaluate subset fitness on fixed inner folds, cache masks, and count requests. |
| `algos.py` | Run GA, BPSO, BDE, NSGA-II, and random search; repair masks and save budget checkpoints. |
| `tracking.py` | Track distinct masks and Pareto archives alongside request counts. |
| `protocols.py` | Implement outer-held-out evaluation and the reporting-practice controls. |
| `filters.py` | Implement ANOVA, random-forest, L1-logistic, and all-feature baselines. |
| `smote.py` | Construct and apply SMOTE sampling plans. |
| `metrics.py` | Calculate balanced accuracy (BA), G-mean, accuracy, F1, MCC, and AUC. |

## Evaluation protocols

`protocols.honest_core` fits standardization on the outer training set and uses its statistics across inner folds. Feature selection uses the outer training data; the selected subset is then evaluated on the held-out outer test set. Strict inner-fold standardization is implemented in `extras/ex_suites6.py` and the additional-data execution code, with entry points listed in the root reproduction guide.

The control protocols implement selection before splitting, SMOTE placement, repeated-measure grouping, and test-based candidate selection. Their outputs measure the reporting practices defined in the archived study protocols. `front_test_ba` is a historical field; the archive-based analysis uses `arch_test_ba`.

## Search budgets and selection

`Wrapper.evaluate(mask)` counts every request, including cache hits. Distinct-mask counts record how many different subsets were evaluated. `Run` saves the incumbent at each requested checkpoint and enforces `KMAX=100` through mask repair. Repair uses the fixed seed `20261008`; algorithm and fold seeds are derived by `splits.rng_for`.

NSGA-II selects the checkpoint candidate by inner error, breaking ties by subset size. The single-objective searches select by their weighted objective. Request-matched, distinct-mask, and size-matched comparisons use the corresponding study entry points.

## Pareto tracking

`tracking.TrackRun` records the nondominated candidates among all distinct masks evaluated up to a checkpoint, using inner error and subset size as objectives. Identical objective vectors retain the first evaluated representative. This archive includes the incumbent used in the archive-based D comparison.

With `archive=True`, `tracking.honest_core_tracked` also records outer scores for archive members to quantify test-based candidate selection. The tracked interface supports ungrouped inner folds; grouped studies use the grouping-aware protocol entry points.

## Extending and checking the implementation

Register a search in `algos.ALGOS` and evaluate Boolean masks through `run.eval(mask)` until `run.done()`. Learner extensions use `learners.SPECS`, `oof_scores`, and `fit_predict`, with score direction and thresholds following the existing interfaces.

From the restored project and its recorded environment:

```sh
python -m unittest discover -s tests
python analysis/test_statsx.py
```

The tests cover data isolation, numerical agreement, search accounting, archive membership, SMOTE behavior, and statistical calculations. Collection-specific tests also check the additional-data and strict-preprocessing pipelines.
