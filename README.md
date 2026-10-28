# ShopMind AI

一个面向求职展示的多模态电商推荐与 AI 导购项目。项目围绕“人、货、场”建立数据分析、召回、排序、解释和评测闭环，而不是只调用大模型生成商品文案。

## 六个可演示模块

1. **用户与业务分析**：行为漏斗、用户分群、价格敏感度、兴趣偏好、复购与流失分析。
2. **推荐基线**：热门推荐、类目热门推荐，为后续算法提供可比较的 baseline。
3. **多路召回与排序**：协同过滤、文本向量、图片向量、热门召回、去重过滤与排序。
4. **AI 导购**：从自然语言提取品类、预算、场景、风格及排除条件，生成有事实依据的推荐理由。
5. **以图搜商品**：使用 CLIP 建立图文共同向量空间，支持相似商品与同风格商品检索。
6. **评测与 Bad Case**：Precision@K、Recall@K、NDCG@K、HitRate@K、覆盖率、P95 延迟及错误分类。

当前里程碑 `M8` 已实现模拟及真实电商数据处理、分析指标、推荐基线、多路召回、加权 RRF、逻辑回归重排、离线发布门禁、中文购物需求理解、约束过滤、可追溯推荐理由、FastAPI 接口、Streamlit 看板和单元测试。

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
```

原始与处理后数据位于被 Git 忽略的目录中，仓库只保存下载、清洗、实验脚本和统计结果，避免重复分发大型数据文件。

首次真实数据对比中，混合召回的 HitRate@10 为 4%，单路模型均为 2%；文本向量召回商品覆盖率为 43.38%。由于当前仅评测 50 名用户，这些结果是可复现的阶段性基线，不代表统计显著的线上收益。

M4 将评测扩大到 200 名用户并离线预计算 ItemCF 相似度。ItemCF P95 延迟由 147.98ms 降至 7.25ms，混合召回 P95 由 185.17ms 降至 44.37ms。文本向量召回取得最高 HitRate@10（7.5%），混合召回取得最高 NDCG@10（0.0287）。

M5 使用前 100 名用户选择融合权重、后 200 名用户独立测试。验证集选择文本向量权重 ×2，测试集 HitRate@10 达到 9.5%、NDCG@10 为 0.0385、覆盖率为 63.13%。失败案例中有 160 个候选未召回、21 个候选已召回但排序未进入 Top-10。

M6 在同一组后 200 名独立测试用户上增加近期兴趣与同购物篮召回并做权重消融。近期兴趣权重 0.25 时，Candidate Recall@50 保持 20%，HitRate@10 从 9.5% 提升至 11.0%，NDCG@10 从 0.0385 提升至 0.0454，P95 延迟从 47.2ms 增至 49.0ms。同购物篮通道在当前切片未带来收益，实验保留但不进入默认组合。

M7 将候选生成与排序拆分，使用 RRF、各召回源排名、商品热度、价格偏差和用户历史长度等 12 个特征训练逻辑回归重排器。独立测试集上重排 HitRate@10 为 10%，低于 M6 的 11%，因此离线发布门禁拒绝上线并继续使用 M6。实验暴露出候选训练集仅 20 个正样本的数据稀疏问题，为后续负采样、更多用户数据和树模型迭代提供依据。

M8 新增离线可复现的中文 AI 导购：抽取品类、价格上下限、场景、风格与排除项，将结构化约束接入推荐链路，并生成包含价格和召回证据的推荐理由。在 6 条人工设计需求、36 个槽位的小规模回归集上，槽位准确率和推荐约束满足率均为 100%；该指标用于功能回归，不作为开放域泛化结论。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/generate_demo_data.py
streamlit run src/shopmind/dashboard.py
```

API：

```bash
uvicorn shopmind.api:app --reload
```

测试：

```bash
pytest
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

