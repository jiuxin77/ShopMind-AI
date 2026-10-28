# ShopMind AI

一个面向求职展示的多模态电商推荐与 AI 导购项目。项目围绕“人、货、场”建立数据分析、召回、排序、解释和评测闭环，而不是只调用大模型生成商品文案。

## 六个可演示模块

1. **用户与业务分析**：行为漏斗、用户分群、价格敏感度、兴趣偏好、复购与流失分析。
2. **推荐基线**：热门推荐、类目热门推荐，为后续算法提供可比较的 baseline。
3. **多路召回与排序**：协同过滤、文本向量、图片向量、热门召回、去重过滤与排序。
4. **AI 导购**：从自然语言提取品类、预算、场景、风格及排除条件，生成有事实依据的推荐理由。
5. **以图搜商品**：使用 CLIP 建立图文共同向量空间，支持相似商品与同风格商品检索。
6. **评测与 Bad Case**：Precision@K、Recall@K、NDCG@K、HitRate@K、覆盖率、P95 延迟及错误分类。

当前里程碑 `M1` 已实现模拟数据生成、分析指标、推荐基线、FastAPI 接口、Streamlit 看板和单元测试。

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

