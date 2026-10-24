# efsbench — held-out evaluation of evolutionary feature selection

纯 numpy / scipy（RF、L1 基线额外用 scikit-learn），Python 3.8+。不需要安装：`PYTHONPATH=src`。

## 它保证什么

1. **诚实协议（`protocols.honest_core`）**：标准化、内层折、全部搜索只用外层训练部分；所选解在搜索结束后按预设检查点评分，外层测试部分不参与选择。内层各折共享外层训练统计量；X1 敏感性研究另行逐内层训练折重拟合。
   单元测试 `tests/test_core.py::...leak...` 把外层测试部分换成乱码，所选特征必须逐位不变。
2. **预算匹配**：每次对包装器的调用都计数（缓存命中也计），所有算法（GA / BPSO / BDE / NSGA-II / 随机搜索）在**同一组检查点**（250 / 1000 / 4000 / 16000 …）
   上取快照。请求数相同不保证不同候选数或计算量相同；比较时也记录不同掩码数。随机搜索是零模型。
3. **可逐位复现**：算法随机数来自 `splits.rng_for(*keys)`（SHA-256 种子）；基数上限修复使用独立的固定种子 `20261008`。没有全局随机状态。
4. **有缺陷协议的对照实现**（用来量化高估，不用于"好成绩"）：`fulldata_core`（先选后分 + 重估）、`smote_outer_core` / `smote_inner_core`（先过采样后划分）、
   `honest_core(groups_tr=...)` 与按组外层划分（重复测量的单元泄漏）、档案上按测试挑解（`tracking` 的 `arch_test_ba`）、多种子取最好。`front_test_ba` 是已废弃 D 定义的历史字段。

## 模块

| 模块 | 内容 |
|---|---|
| `datasets.py` | OpenML / UCI 公开数据的 md5 校验下载、`load(name) -> X, y, groups`（少数类 = 1） |
| `splits.py` | `rng_for`、分层 / 分层分组 K 折、`outer_splits` |
| `learners.py` | 固定超参的 5-NN 与类平衡加权岭分类（不调参，避免多一层选择） |
| `wrapper.py` | `Wrapper.evaluate(mask)`：内层折上的 1 − BA（或 acc / G-mean / AUC），精确计数 |
| `algos.py` | `Run`（增量最优、检查点快照、基数上限 `KMAX=100` 的就地修复）、五个搜索算法 |
| `protocols.py` | 诚实协议、有缺陷协议、SMOTE 协议 |
| `filters.py` | ANOVA-F top-k（k 由内层 CV 选）、RF 重要性 top-k、L1 逻辑回归、全部特征 |
| `smote.py` | SMOTE（可只对训练部分施加） |
| `metrics.py` | BA、G-mean、acc、F1、MCC、AUC（平均秩） |

## 最小用法

```python
from efsbench import datasets as D, protocols as P, splits as S
X, y, g = D.load("colon")
tr, te = S.make_folds(y, 5, S.rng_for("demo"), g)[0]
out = P.honest_core(X[tr], y[tr], X[te], y[te], "ga", "ridge", ("inner", 0), ("algo", 0), checkpoints=(1000, 4000))
out[4000]["test_ba"], 1 - out[4000]["inner_err"]     # 诚实 BA vs 适应度（论文里常被当成"性能"报告的那个数）
```

完整例子：`examples/quickstart.py`。

## 加入自己的算法 / 学习器

* 算法：写 `def my_algo(run, rng, init_p): ...`，循环调用 `err, k, f = run.eval(mask)`（`mask` 为布尔向量），直到 `run.done()`；
  在 `algos.ALGOS` 注册即可。基数上限修复、评估计数、检查点快照都在 `Run.eval` 里，算法无需关心。
* 学习器：在 `learners.SPECS` 加一项，并在 `oof_scores` / `fit_predict` 里加分支（折外分数越大越像少数类；`threshold(spec)` 给出判决阈值）。

## 请求数、不同候选数与 Pareto 档案（`tracking.py`，评审后新增）

`tracking.TrackRun` 是 `algos.Run` 的即插即用替代品（不额外抽随机数，轨迹逐位相同，见 `tests/test_tracking.py`），额外提供：
检查点既可按**包装器请求数**也可按**不同掩码数**计；设 `archive=True` 时，每个检查点返回 **Pareto 档案**（截至该点全部已评估候选的非支配集，目标向量相同的合并为首个评估的代表，当前最优解必在其中，故 D ≥ 0）；
并记录请求数与不同掩码数。`tracking.honest_core_tracked` 使用相同外层测试隔离流程；设 `archive=True` 时还在外层测试部分上评估每个档案成员（用于"按测试集挑 Pareto 解"的高估 D）。该接口不支持 `groups_tr`，不能直接用于需要分组内层折的数据。
第一稿里 D 的旧定义（当前最优解 + 上一代发布的种群前沿）已废弃，见 `docs/DEVIATIONS.md` #7。

## 测试

```
python -m unittest discover -s tests   # 25 个单元测试：泄漏探测（含 Pareto 档案）、与 sklearn 的数值等价、预算计数、SMOTE 性质、TrackRun 与 Run 逐位相同
python analysis/test_statsx.py         # 分析用统计函数的测试（Holm、Clopper–Pearson、精确期望最大值、双层自助、t 区间 / 聚类自助）
```
