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
    st.caption("当前已实现热门、ItemCF、BM25 与 TF-IDF 文本向量四路召回；图片向量将在后续接入。")
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
    st.info("Bad Case 分类面板将在保存逐用户评测明细后加入。")
