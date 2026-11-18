
# ShopMind AI

[在线 Demo](https://dashboardpy-atdyds5ymth55g6m623f5z.streamlit.app/) · [个人作品集](https://jiuxin77.github.io/) · [架构说明](docs/system_architecture.md)

**Jiuxin (Harriet) Xiao · 独立个人项目**

## 三分钟体验与指标口径

1. 查看用户行为漏斗，再体验多路召回和 AI 导购，检查推荐来源与约束。
2. 在评测模块比较 M6 和未获门禁通过的 M7，查看失败案例。
3. 体验图片检索：当前是合成缩略图上的特征基线，不是 CLIP 或真实照片语义检索。

| 范围 | 口径 |
| --- | --- |
| 在线演示 | 300 个合成用户、200 个商品、5,000 条演示行为 |
| 离线实验 | 单独准备的 UCI Online Retail 数据和实验用户切片 |
| M6 对比 | 同一 200 用户切片；HitRate@10：9.5% → 11.0%（+1.5 个百分点）；NDCG@10：0.0385 → 0.0454 |
| 候选与延迟 | Candidate Recall@50 均为 20%；P95：47.2 → 49.0 ms，指离线召回测量，不是网页时延 |
| M7 重排 | HitRate@10 为 10%，低于保留基线的 11%，未通过离线门禁 |
| 自动化检查 | M10 保存记录为 15 项通过，并非访问页面时现场重跑 |

后续实验反复比较了同一测试用户切片，结果属于探索性离线对比，不是完全未参与选型的最终盲测，也不是线上业务提升。下一轮应冻结方案并使用新的时间切分留出集。当前导购由规则与检索驱动，大模型 Agent 和生产级监控属于后续方向。

独立构建端到端电商推荐系统，包含用户与业务分析、推荐基线、多路召回与融合排序、自然语言导购、以图搜商品、离线评测与 Bad Case 分析六个演示模块。

实现热门、ItemCF、BM25、TF-IDF、近期兴趣和购物篮等召回策略，通过加权 RRF 融合候选，并使用 HitRate@10、NDCG@10、覆盖率和 P95 延迟评估效果；建立自动化测试与离线模型发布门禁。

在线 Demo 使用 300 名合成用户、200 个商品和 5,000 条演示交互；已保存的检查记录显示 15 项测试通过。导购目前基于规则与检索，图像检索为合成缩略图上的轻量特征基线。

## 六个可演示模块

1. **用户与业务分析**：行为漏斗、用户分群、价格敏感度、兴趣偏好、复购与流失分析。
2. **推荐基线**：热门推荐、类目热门推荐，为后续算法提供可比较的 baseline。
3. **多路召回与排序**：协同过滤、文本向量、图片向量、热门召回、去重过滤与排序。
4. **AI 导购**：从自然语言提取品类、预算、场景、风格及排除条件，生成有事实依据的推荐理由。
5. **以图搜商品**：使用轻量视觉特征完成演示图相似检索，并为后续真实图片与 CLIP 接入保留评测基线。
6. **评测与 Bad Case**：Precision@K、Recall@K、NDCG@K、HitRate@K、覆盖率、P95 延迟及错误分类。

当前里程碑 `M11` 已完成六个可演示模块、可复现实验、系统架构、简历与面试材料、一键启动、GitHub Actions CI 和 Streamlit Cloud 部署配置。

### M2 召回链路

```text
热门召回 ─┐
ItemCF  ──┼─> Reciprocal Rank Fusion ─> 业务规则过滤 ─> Top-K
BM25    ──┘
```

评测采用按用户最后一次正向行为留出的 Leave-One-Out 方法，输出 Precision@K、Recall@K、HitRate@K、NDCG@K、商品覆盖率、平均延迟与 P95 延迟。

## 真实数据实验

M3 使用 UCI Machine Learning Repository 的 Online Retail 数据集。该数据包含 2010—2011 年英国线上零售商的 541,909 条交易记录，许可为 CC BY 4.0。

```bash
python scripts/prepare_uci_online_retail.py --max-users 500 --max-products 1000
python scripts/run_retrieval_experiment.py --users 50 --k 10
python scripts/build_item_cf_index.py
python scripts/tune_fusion_weights.py --validation-users 100 --test-users 200
python scripts/run_recall_expansion_experiment.py
python scripts/run_ranking_experiment.py
python scripts/evaluate_shopping_assistant.py
python scripts/evaluate_visual_search.py
```

原始与处理后数据位于被 Git 忽略的目录中，仓库只保存下载、清洗、实验脚本和统计结果，避免重复分发大型数据文件。

首次真实数据对比中，混合召回的 HitRate@10 为 4%，单路模型均为 2%；文本向量召回商品覆盖率为 43.38%。由于当前仅评测 50 名用户，这些结果是可复现的阶段性基线，不代表统计显著的线上收益。

M4 将评测扩大到 200 名用户并离线预计算 ItemCF 相似度。ItemCF P95 延迟由 147.98ms 降至 7.25ms，混合召回 P95 由 185.17ms 降至 44.37ms。文本向量召回取得最高 HitRate@10（7.5%），混合召回取得最高 NDCG@10（0.0287）。

M5 使用前 100 名用户选择融合权重、后 200 名用户独立测试。验证集选择文本向量权重 ×2，测试集 HitRate@10 达到 9.5%、NDCG@10 为 0.0385、覆盖率为 63.13%。失败案例中有 160 个候选未召回、21 个候选已召回但排序未进入 Top-10。

M6 在同一组后 200 名独立测试用户上增加近期兴趣与同购物篮召回并做权重消融。近期兴趣权重 0.25 时，Candidate Recall@50 保持 20%，HitRate@10 从 9.5% 提升至 11.0%，NDCG@10 从 0.0385 提升至 0.0454，P95 延迟从 47.2ms 增至 49.0ms。同购物篮通道在当前切片未带来收益，实验保留但不进入默认组合。

M7 将候选生成与排序拆分，使用 RRF、各召回源排名、商品热度、价格偏差和用户历史长度等 12 个特征训练逻辑回归重排器。独立测试集上重排 HitRate@10 为 10%，低于 M6 的 11%，因此离线发布门禁拒绝上线并继续使用 M6。实验暴露出候选训练集仅 20 个正样本的数据稀疏问题，为后续负采样、更多用户数据和树模型迭代提供依据。

M8 新增离线可复现的中文 AI 导购：抽取品类、价格上下限、场景、风格与排除项，将结构化约束接入推荐链路，并生成包含价格和召回证据的推荐理由。在 6 条人工设计需求、36 个槽位的小规模回归集上，槽位准确率和推荐约束满足率均为 100%；该指标用于功能回归，不作为开放域泛化结论。

M9 新增轻量级视觉检索工程基线：为演示商品生成确定性合成缩略图，提取亮度不变的色度比例、Hue/Saturation 直方图、空间颜色和边缘特征，并使用余弦相似度检索。在 50 张、0.72–1.28 倍亮度扰动的合成图查询中 Top-1 与 Top-5 均为 100%。该结果只验证演示图亮度鲁棒性，不代表真实照片或 CLIP 语义检索能力。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/generate_demo_data.py
streamlit run src/shopmind/dashboard.py
```

完成安装后也可以一键启动：

```bash
python scripts/run_demo.py
```

API：

```bash
uvicorn shopmind.api:app --reload
```

测试：

```bash
pytest
python scripts/release_check.py
python scripts/deployment_check.py
```

## 项目结构

```text
ShopMind-AI/
├── data/                  # 演示数据（可替换为公开电商数据集）
├── docs/                  # 产品、指标和实验设计
├── scripts/               # 数据生成与批处理脚本
├── src/shopmind/          # 分析、推荐、API 与可视化
└── tests/                 # 自动化测试
```

## 简历指标原则

README 中只记录脚本真实产出的结果。禁止预先填写“提升 30%”等未经实验验证的数据。



