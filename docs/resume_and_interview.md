# 简历与面试表达

## 中文简历版本

**ShopMind AI｜电商推荐与智能导购系统（个人项目）**

- 基于 Python、Pandas、scikit-learn、FastAPI 和 Streamlit 搭建电商推荐闭环，覆盖人货场分析、多路召回、排序融合、AI 导购、视觉检索、离线评测与 Bad Case 分析。
- 清洗并抽样 UCI Online Retail 541,909 条真实交易记录，构建热门、ItemCF、近期兴趣、BM25 与 TF-IDF 多路召回，通过加权 RRF 将独立测试集 HitRate@10 从 9.5% 提升至 11.0%，NDCG@10 从 0.0385 提升至 0.0454。
- 将 ItemCF 相似度离线预计算，使其 P95 延迟从 147.98ms 降至 7.25ms；建立模型发布门禁，识别并拒绝未超过基线的逻辑回归重排模型。
- 实现中文购物需求结构化抽取和事实可追溯推荐理由，在 6 条、36 个槽位的功能回归集上取得 100% 槽位准确率和约束满足率，并明确限制其非开放域指标。
- 建立 15 项自动化测试、Ruff 静态检查和一键发布检查；使用 Streamlit 展示业务漏斗、消融实验、Bad Case、导购和轻量视觉检索结果。

## English résumé version

**ShopMind AI | E-commerce Recommendation & Shopping Assistant**

- Built an end-to-end recommendation portfolio using Python, Pandas, scikit-learn, FastAPI, and Streamlit, covering analytics, multi-channel retrieval, ranking, explainability, evaluation, and visual search.
- Processed the 541,909-row UCI Online Retail dataset and combined popularity, ItemCF, recent-interest, BM25, and TF-IDF retrieval with weighted RRF, improving HitRate@10 from 9.5% to 11.0% and NDCG@10 from 0.0385 to 0.0454 on a fixed 200-user test split.
- Precomputed ItemCF similarities to reduce P95 latency from 147.98 ms to 7.25 ms, and implemented an offline release gate that rejected a reranker when it failed to beat the production baseline.
- Implemented deterministic Chinese shopping-intent extraction and evidence-grounded explanations, achieving 100% slot and constraint accuracy on a clearly scoped 36-slot regression suite.
- Added 15 automated tests, static analysis, reproducible experiment artifacts, and an interactive six-module dashboard.

## 面试讲解顺序

1. 先讲业务目标：帮助用户在预算、场景和偏好约束下更快找到商品。
2. 展示人货场分析，再说明为什么需要热门、协同、文本和近期兴趣多路召回。
3. 展示 M5→M6 的同切片消融结果，并解释为什么购物篮通道没有上线。
4. 展示 M7 发布门禁：模型更复杂不等于更好，离线指标下降就拒绝发布。
5. 演示 AI 导购和推荐证据，最后说明视觉模块目前只是工程基线及下一步计划。
