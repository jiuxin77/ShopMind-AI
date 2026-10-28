import plotly.express as px
import streamlit as st

from shopmind.analytics import behavior_funnel, category_performance, price_sensitivity, scene_performance
from shopmind.data import load_demo_data
from shopmind.recommender import popular_recommend


st.set_page_config(page_title="ShopMind AI", layout="wide")
st.title("ShopMind AI｜多模态商品推荐与 AI 导购")
st.caption("从人、货、场分析到召回、排序、解释与 Bad Case 的完整推荐闭环")

try:
    users, products, interactions = load_demo_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

tabs = st.tabs(["1 用户与业务分析", "2 推荐基线", "3 多路召回", "4 AI 导购", "5 以图搜商品", "6 评测与 Bad Case"])

with tabs[0]:
    a, b, c, d = st.columns(4)
    a.metric("用户数", f"{len(users):,}")
    b.metric("商品数", f"{len(products):,}")
    c.metric("行为数", f"{len(interactions):,}")
    d.metric("购买数", f"{(interactions.event_type == 'purchase').sum():,}")
    left, right = st.columns(2)
    funnel = behavior_funnel(interactions)
    left.plotly_chart(px.funnel(funnel, x="count", y="event_type", title="用户行为漏斗"), use_container_width=True)
    sensitivity = price_sensitivity(interactions, products)
    right.plotly_chart(px.bar(sensitivity, x="price_band", y="purchase_rate", title="价格带购买率"), use_container_width=True)
    left2, right2 = st.columns(2)
    category = category_performance(interactions, products)
    left2.plotly_chart(px.bar(category, x="category", y="conversion_rate", color="refund_rate", title="品类转化与退款"), use_container_width=True)
    scene = scene_performance(interactions)
    right2.plotly_chart(px.bar(scene, x="scene", y="conversion_rate", title="场景转化率"), use_container_width=True)

with tabs[1]:
    category_options = ["全部"] + sorted(products["category"].unique().tolist())
    selected = st.selectbox("品类", category_options)
    budget = st.slider("最高预算", 50, 2000, 600, 50)
    result = popular_recommend(products, interactions, None if selected == "全部" else selected, budget)
    st.dataframe(result[["product_id", "title", "category", "brand", "price", "rating", "score", "reason"]], use_container_width=True)

with tabs[2]:
    st.info("M2：接入 ItemCF、BM25、文本向量、图片向量四路召回，并展示各路候选和融合分数。")
with tabs[3]:
    st.info("M3：接入结构化需求抽取，处理品类、预算、场景、风格和排除条件，并生成事实可追溯的推荐理由。")
with tabs[4]:
    st.info("M4：接入 CLIP 图文向量，支持上传图片查找相似商品及图文一致性检查。")
with tabs[5]:
    st.info("M5：展示 Precision@K、Recall@K、NDCG@K、HitRate@K、覆盖率、平均延迟、P95 及 Bad Case 分类。")

