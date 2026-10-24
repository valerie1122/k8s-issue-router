# k8s Issue Router — 一周计划 v2（2026-10-22 → 10-28）

## 一句话
kubernetes/kubernetes 每个新 issue 进来，判断该归哪个 SIG；有把握就自动分派，没把握交给人。离线回测 + 线上影子模式（只记录预测，不真打标签）验证。

## 简历要回答的那一句话（所有 eval 都为它服务）
> 在 ≥95% precision 下自动分派 X% 的新 issue；比只用大模型便宜 Y 倍；分派用时 Z 秒，人工中位数 W 小时；线上影子模式处理 N 条，precision 与离线差距 ±d。

## v1 → v2 改了什么
1. **问题简化成“选主 SIG”**：主指标是 top-1 预测是否在标准答案集合里（真实分派也是先给一个 SIG，那个 SIG 再拉别人）。多标签 F1 降为次要指标。一周内做得完，也更贴近真实流程。
2. **加“不分派”这一类**：很多 issue 是提问/支持类，最后不归任何 SIG。模型可以判“不该分派”，这一类也计入覆盖率。
3. **作者自己指定了 SIG 的都排除**：除了 `/sig`，还有 `@kubernetes/sig-xxx-bugs` 这种 @ 团队的写法，都算“作者已分派”，不进目标集合。
4. **原始 body**：用 GraphQL `userContentEdits` 取最早版本，还原创建时的内容；拿不到的打标记，单独统计。
5. **回放回测（replay）**：线上一周只有约 50 条，而且很多到周三还没人工标签，样本太小。所以把影子服务的同一套代码，按时间顺序在最近 2 个月的 issue 上“重放”一遍，得到大样本线上口径数字。真实线上数据作为小样本佐证，附置信区间。
6. **置信区间**：所有 precision 都报 Wilson 95% 区间，样本小的时候不夸大。
7. **多一个便宜基线**：向量 + 逻辑回归（10 分钟能做完）。如果它和 ModernBERT 差不多，这本身就是一个发现，要写进去。
8. **三层级联**：ModernBERT → Haiku 级模型 → Sonnet 级 agent → 人。成本曲线更好看，每层的取舍都能讲。
9. **`lookup_owners` 改成组件映射**：issue 里很少写代码路径，常写的是组件名（kubelet、kube-proxy、scheduler）。从 `sigs.yaml` 子项目 + OWNERS 建“组件 → SIG”表。
10. **阈值最后定**：先画 precision-覆盖率曲线，同时报 90% 和 95% 两个点，不提前锁死 95%。

## 技术栈
uv · Pydantic · Postgres + pgvector（Neon）· sentence-transformers · ModernBERT（Modal GPU）· Claude API（tool use、structured outputs、prompt caching、Batch API）· MCP server（三个工具）· Langfuse（tracing）· Modal cron（影子服务）· Streamlit（面板）· Inspect AI（可选）

## 每日计划
| 天 | 内容 | 完成标准 |
|---|---|---|
| D1 周四晚 | GitHub token；拉 6 个月；统计；后台开始拉 3 年 | 统计表：周量、目标集合占比、多 SIG 占比、人工用时 |
| D2 周五 | `sigs.yaml` 解析 + 组件映射；标准答案；时间切分；eval 函数（曲线、Wilson、ECE）；B1 相似投票、B1.5 向量+逻辑回归、B2 零样本 | 第一张结果表 |
| D3 周六 | ModernBERT 微调 + 校准；影子服务（轮询、幂等、重试、replay 模式、存库）；部署到 Modal，只跑 M2 | 线上开始攒数据 |
| D4 周日 | MCP server（三个工具）；M1 agent；Batch API 跑离线 eval | M1 离线结果 + 成本 |
| D5 周一 | 级联 + 选阈值；replay 回测；测成本和延迟；级联上线；Langfuse | 完整对比表 |
| D6 周二 | 错误分析 30 条；面板；README + 架构图 | repo 可以公开 |
| D7 周三 | 线上数字快照（带区间）；简历 bullet；面试取舍题（中英）；博客草稿 | 简历可用 |

## 落后时先砍（按顺序）
Inspect AI → Streamlit 面板（改成 README 里的静态图）→ Langfuse（改成自己记日志）→ Haiku 那一层 → 博客。
**不能砍：** 时间切分的离线表、级联、replay 回测、影子服务在跑、README。

## 周三之后
影子服务继续跑满 2–3 周，更新线上数字；正式博客；联系 SIG Contributor Experience。

## 不做的事
不训大模型、不做强化学习、不扩到别的 repo、不真的往 k8s 打标签或发评论。

## D1 数据结论（10-22）
- 3 年 5968 个 issue（已去掉 PR），每周中位数 36 个；最近 6 个月 989 个，每周 35 个。量够用，暂时不加 pytorch。
- 作者自己指定 SIG 是主流：最近 6 个月里 52% 在正文写了 `/sig`，另有约 21% 在建完 2 分钟内补了 `/sig` 评论。正文里写的 SIG 96% 到第 30 天还在 → 作者自己选的标签可以当训练数据。
- 真正由别人分派的只占约 10%（6 个月 92 个），人工用时中位数 12.6 小时，p75 56 小时。这才是“比人工快多少”的基线。
- 没写 `/sig` 的 issue 里 26% 过了 30 天还没有 SIG（6 个月 110 个，其中 45 个还开着，大多是 kind/bug）。这是最直接的痛点。
- 多 SIG 占 16%；sig/node + api-machinery 占约 60%；有 10 个 SIG 在 6 个月里少于 10 条 → 必须报 macro 指标。

### 对计划的影响
- 训练和主测试集用所有带 SIG 的 issue（删掉 `/sig` 命令），任务是“在创建那一刻预测 SIG”。
- “别人分派”子集单独报告，因为最近 2 个月只有约 30 条，所以这个子集用最近 6 个月做测试，附置信区间。
- 产品叙事：建 issue 时就给出 SIG → 覆盖那 26% 一直没人管的，和那 10% 要等半天的。
- 线上每天约 5 条新 issue，一周大约 35 条。

## D2 进度（10-22 晚，提前开工）
- 还原了原始文本：767 个标题、2170 个正文后来被改过；102 个是后来才在正文补的 `/sig`。数据集一律用创建时的版本（`fetch_originals.py`）。
- `sigs.yaml` 有 24 个 SIG，3 年里出现过的标签全部能对上，不需要做改名映射。
- 数据集 `data/processed/dataset.jsonl`：train 4199 / val 356 / test 441 条有标准答案；test 里还有 170 条不满 30 天（留给线上对比）。
- 清洗：去掉 HTML 注释、行首和行内的 `/sig /kind /area ...`，以及 `@kubernetes/sig-xxx`；testgrid 链接、KEP 路径保留（建 issue 时就有的真实线索）。
- 第一张表（`RESULTS_baselines.txt`）：TF-IDF + 逻辑回归 top-1 0.75，比全猜 node（0.40）高很多；kNN 0.68。**这个便宜基线就是要打败的对象。**
- 注意：表里的 cov@90/95 是用同一个集合定的阈值（偏乐观）。正式数字要在 val 上定阈值，再拿到 test 上用（D5）。

## 环境限制
- 这里连不上 Hugging Face / Modal / Langfuse / Neon；能连 GitHub、PyPI、Anthropic。
- 处理：允许列表加域名（待 Valerie 决定）；影子服务改用 GitHub Actions 定时任务。

## 每日复盘：10-23（周五）
**做完：** D2 全部完成（向量基线、Haiku 零样本、抽查 20 条）；D3 的 ModernBERT 微调和校准提前完成；级联试算。比计划快约 1 天。结果见 `RESULTS.md`。
**做得不好：**
1. 没在开工前测所有外部服务能不能连通，环境问题来回折腾了约 1.5 小时 → 以后每天开工先跑连通性检查。
2. 说错了两件事（key 格式、误判 Modal 训练失败）→ 先查证再下结论。
3. eval 里失败的请求被跳过了，没算作“交给人”，B2 数字偏高。
4. ModernBERT 只训了一个随机种子，不知道波动有多大。
5. 还没验证 Haiku 有没有见过测试集。
**调整：**
- D3：修 eval；补跑 30 个失败请求；ModernBERT 训 3 个种子；按 Haiku 训练截止日期前后拆开做污染检查；影子服务上 GitHub Actions。
- D4：agent 只处理 ModernBERT 和 Haiku 意见不一致的约 17%，以及可能不该分派的 issue。
- 流程：需要 Valerie 在终端里跑的步骤，每天合并成一个命令或一份清单。
