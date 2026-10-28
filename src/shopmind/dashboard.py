import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from shopmind.analytics import (
    behavior_funnel,
    category_performance,
    price_sensitivity,
    scene_performance,
)
from shopmind.data import load_demo_data
from shopmind.evaluation import evaluate_hybrid
from shopmind.recommender import popular_recommend
from shopmind.retrieval import hybrid_recommend

st.set_page_config(page_title="ShopMind AI", layout="wide")
st.title("ShopMind AI｜多模态商品推荐与 AI 导购")
st.caption("从人、货、场分析到召回、排序、解释与 Bad Case 的完整推荐闭环")

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"

try:
    users, products, interactions = load_demo_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

tabs = st.tabs(
    [
        "1 用户与业务分析",
        "2 推荐基线",
        "3 多路召回",
        "4 AI 导购",
        "5 以图搜商品",
        "6 评测与 Bad Case",
    ]
)

with tabs[0]:
    a, b, c, d = st.columns(4)
    a.metric("用户数", f"{len(users):,}")
    b.metric("商品数", f"{len(products):,}")
    c.metric("行为数", f"{len(interactions):,}")
    d.metric("购买数", f"{(interactions.event_type == 'purchase').sum():,}")
    left, right = st.columns(2)
    funnel = behavior_funnel(interactions)
    left.plotly_chart(
        px.funnel(funnel, x="count", y="event_type", title="用户行为漏斗"), use_container_width=True
    )
    sensitivity = price_sensitivity(interactions, products)
    right.plotly_chart(
        px.bar(sensitivity, x="price_band", y="purchase_rate", title="价格带购买率"),
        use_container_width=True,
    )
    left2, right2 = st.columns(2)
    category = category_performance(interactions, products)
    left2.plotly_chart(
        px.bar(
            category, x="category", y="conversion_rate", color="refund_rate", title="品类转化与退款"
        ),
        use_container_width=True,
    )
    scene = scene_performance(interactions)
    right2.plotly_chart(
        px.bar(scene, x="scene", y="conversion_rate", title="场景转化率"), use_container_width=True
    )

with tabs[1]:
    category_options = ["全部"] + sorted(products["category"].unique().tolist())
    selected = st.selectbox("品类", category_options)
    budget = st.slider("最高预算", 50, 2000, 600, 50)
    result = popular_recommend(
        products, interactions, None if selected == "全部" else selected, budget
    )
    st.dataframe(
        result[["product_id", "title", "category", "brand", "price", "rating", "score", "reason"]],
        use_container_width=True,
    )

with tabs[2]:
    st.subheader("多路召回与 Reciprocal Rank Fusion")
    recall_left, recall_right = st.columns(2)
    user_id = recall_left.selectbox("用户", users["user_id"].head(100).tolist())
    query = recall_right.text_input("搜索需求", "通勤 简约")
    recall_category = recall_left.selectbox(
        "召回品类", ["全部"] + sorted(products["category"].unique().tolist())
    )
    recall_budget = recall_right.number_input(
        "预算上限", min_value=50, max_value=3000, value=800, step=50
    )
    hybrid = hybrid_recommend(
        products,
        interactions,
        user_id=user_id,
        query=query,
        category=None if recall_category == "全部" else recall_category,
        budget_max=float(recall_budget),
        top_k=10,
    )
    st.dataframe(
        hybrid[
            ["product_id", "title", "category", "brand", "price", "score", "evidence", "reason"]
        ],
        use_container_width=True,
    )
    st.caption(
        "当前已实现热门、ItemCF、近期兴趣、同购物篮、BM25 与 TF-IDF 文本向量召回；"
        "离线实验选择有效通道进入默认组合。"
    )
with tabs[3]:
    st.info(
        "M3：接入结构化需求抽取，处理品类、预算、场景、风格和排除条件，并生成事实可追溯的推荐理由。"
    )
with tabs[4]:
    st.info("M4：接入 CLIP 图文向量，支持上传图片查找相似商品及图文一致性检查。")
with tabs[5]:
    st.subheader("离线评测与性能指标")
    eval_users = st.slider("评测用户数", 10, 100, 30, 10)
    if st.button("运行 Leave-One-Out 评测"):
        with st.spinner("正在评测…"):
            metrics = evaluate_hybrid(products, interactions, k=10, max_users=eval_users)
        metric_columns = st.columns(4)
        metric_columns[0].metric("HitRate@10", f"{metrics['hit_rate_at_k']:.2%}")
        metric_columns[1].metric("NDCG@10", f"{metrics['ndcg_at_k']:.3f}")
        metric_columns[2].metric("商品覆盖率", f"{metrics['catalog_coverage']:.2%}")
        metric_columns[3].metric("P95 延迟", f"{metrics['p95_latency_ms']:.1f} ms")
        st.json(metrics)
    comparison_path = ARTIFACT_DIR / "m4_model_comparison.json"
    bad_case_path = ARTIFACT_DIR / "m4_bad_cases.csv"
    if comparison_path.exists():
        comparison = pd.DataFrame(json.loads(comparison_path.read_text(encoding="utf-8")))
        st.plotly_chart(
            px.bar(
                comparison,
                x="model",
                y=["hit_rate_at_k", "ndcg_at_k", "catalog_coverage"],
                barmode="group",
                title="真实数据模型对比",
            ),
            use_container_width=True,
        )
        st.dataframe(comparison, use_container_width=True)
    if bad_case_path.exists():
        details = pd.read_csv(bad_case_path)
        bad_cases = details[details["hit"] == False]
        st.metric("未命中 Bad Case", f"{len(bad_cases)} / {len(details)}")
        st.dataframe(bad_cases.head(100), use_container_width=True)
    m5_path = ARTIFACT_DIR / "m5_weight_tuning.json"
    m5_bad_case_path = ARTIFACT_DIR / "m5_bad_cases.csv"
    if m5_path.exists():
        m5 = json.loads(m5_path.read_text(encoding="utf-8"))
        st.subheader("加权融合独立测试结果")
        weighted = m5["test_metrics"]
        weighted_columns = st.columns(4)
        weighted_columns[0].metric("HitRate@10", f"{weighted['hit_rate_at_k']:.2%}")
        weighted_columns[1].metric("NDCG@10", f"{weighted['ndcg_at_k']:.4f}")
        weighted_columns[2].metric("覆盖率", f"{weighted['catalog_coverage']:.2%}")
        weighted_columns[3].metric("P95", f"{weighted['p95_latency_ms']:.1f} ms")
        st.caption(f"验证集选出的权重：{m5['best_weights']}")
    if m5_bad_case_path.exists():
        m5_details = pd.read_csv(m5_bad_case_path)
        distribution = (
            m5_details["bad_case_type"]
            .fillna("hit")
            .value_counts()
            .rename_axis("type")
            .reset_index(name="count")
        )
        st.plotly_chart(
            px.pie(distribution, names="type", values="count", title="Bad Case 类型分布"),
            use_container_width=True,
        )
    m6_path = ARTIFACT_DIR / "m6_recall_expansion.json"
    if m6_path.exists():
        m6 = json.loads(m6_path.read_text(encoding="utf-8"))
        m6_results = pd.DataFrame(m6["results"])
        st.subheader("M6 召回通道消融实验")
        base = m6_results.loc[m6_results["model"] == "m5_weighted"].iloc[0]
        best = m6_results.loc[m6_results["model"] == m6["best_model"]].iloc[0]
        m6_columns = st.columns(4)
        m6_columns[0].metric(
            "HitRate@10",
            f"{best['hit_rate_at_k']:.2%}",
            f"{best['hit_rate_at_k'] - base['hit_rate_at_k']:+.2%}",
        )
        m6_columns[1].metric(
            "NDCG@10",
            f"{best['ndcg_at_k']:.4f}",
            f"{best['ndcg_at_k'] - base['ndcg_at_k']:+.4f}",
        )
        m6_columns[2].metric("Candidate Recall@50", f"{best['candidate_recall']:.2%}")
        m6_columns[3].metric(
            "P95 延迟",
            f"{best['p95_latency_ms']:.1f} ms",
            f"{best['p95_latency_ms'] - base['p95_latency_ms']:+.1f} ms",
            delta_color="inverse",
        )
        st.caption(
            f"固定独立测试集：{m6['test_slice']}；最优配置：{m6['best_model']}。"
            "购物篮召回在该切片未带来收益，因此未进入默认组合。"
        )
        st.plotly_chart(
            px.bar(
                m6_results,
                x="model",
                y=["hit_rate_at_k", "candidate_recall"],
                barmode="group",
                title="扩展召回通道消融对比",
            ),
            use_container_width=True,
        )
