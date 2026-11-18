import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from shopmind.analytics import (
    behavior_funnel,
    category_performance,
    price_sensitivity,
    scene_performance,
)
from shopmind.data import load_demo_data
from shopmind.evaluation import evaluate_hybrid
from shopmind.llm_shopping_agent import create_openai_agent
from shopmind.recommender import popular_recommend
from shopmind.retrieval import hybrid_recommend
from shopmind.visual_search import build_visual_index, visual_search

st.set_page_config(page_title="ShopMind AI", layout="wide")
st.title("ShopMind AI | Multimodal Product Recommendation and AI Shopping")
st.caption("An end-to-end recommendation workflow spanning users, products, context, retrieval, ranking, explanations, and bad-case analysis")

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
IMAGE_DIR = Path(__file__).resolve().parents[2] / "data" / "generated_product_images"


@st.cache_data(show_spinner=False)
def cached_demo_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the demo dataset once per deployed app process."""
    return load_demo_data()


@st.cache_resource(show_spinner="Building the visual search index for the first time…")
def cached_visual_index(products_for_index: pd.DataFrame):
    """Build synthetic thumbnails and their embeddings only when this module is opened."""
    return build_visual_index(products_for_index, IMAGE_DIR)


try:
    users, products, interactions = cached_demo_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

CATEGORY_EN = {"女装": "Women's Fashion", "家居": "Home", "护肤": "Skincare", "数码": "Electronics", "食品": "Food"}
TITLE_EN = {
    "休闲针织衫": "Casual Knitwear",
    "低糖麦片": "Low-Sugar Cereal",
    "即食燕麦": "Instant Oats",
    "坚果组合": "Mixed Nuts",
    "床品套装": "Bedding Set",
    "护眼台灯": "Eye-Care Desk Lamp",
    "无线耳机": "Wireless Headphones",
    "机械键盘": "Mechanical Keyboard",
    "清爽精华": "Refreshing Serum",
    "温和洁面": "Gentle Cleanser",
    "移动电源": "Portable Charger",
    "简约半身裙": "Minimalist Skirt",
    "舒缓面霜": "Soothing Face Cream",
    "通勤衬衫": "Commute Shirt",
    "香薰机": "Aroma Diffuser",
}
BRAND_EN = {"云间": "Cloudlane", "北岸": "Northshore", "星野": "Xingye", "简物": "Simply", "青禾": "Greenfield"}
SCENE_EN = {"宿舍": "Dorm", "旅行": "Travel", "日常": "Everyday", "运动": "Fitness", "送礼": "Gifting", "通勤": "Commute"}


def display_products(frame: pd.DataFrame) -> pd.DataFrame:
    """Translate synthetic catalog labels for the public English demo."""
    display = frame.copy()
    for column, mapping in (("title", TITLE_EN), ("category", CATEGORY_EN), ("brand", BRAND_EN)):
        if column in display.columns:
            display[column] = display[column].replace(mapping)
    return display

overview = st.columns(5)
overview[0].metric("Milestone", "M12")
overview[1].metric("Raw Transactions", "541,909")
overview[2].metric("Holdout HitRate@10", "11.0%")
overview[3].metric("ItemCF P95", "7.25 ms")
overview[4].metric("Automated Tests", "21")

with st.expander("View system architecture and data flow"):
    architecture_labels = [
        "Transaction and product data",
        "User-product-context analysis",
        "Multi-channel retrieval",
        "Weighted RRF",
        "Business constraint filtering",
        "Top-K recommendations",
        "AI shopping explanations",
        "Offline evaluation",
        "Release gate",
    ]
    architecture = go.Figure(
        go.Sankey(
            node={"label": architecture_labels, "pad": 18, "thickness": 18},
            link={
                "source": [0, 0, 2, 3, 4, 5, 5, 7],
                "target": [1, 2, 3, 4, 5, 6, 7, 8],
                "value": [2, 8, 8, 8, 8, 4, 4, 4],
            },
        )
    )
    architecture.update_layout(height=330, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    st.plotly_chart(architecture, use_container_width=True)
    st.caption(
        "Candidate retrieval, rank fusion, business filtering, explanations, and evaluation are decoupled. Models that fail the release gate never replace the default strategy."
    )

pages = [
    "1 User and Business Analytics",
    "2 Recommendation Baseline",
    "3 Multi-Channel Retrieval",
    "4 AI Shopping Agent",
    "5 Visual Product Search",
    "6 Evaluation and Bad Cases",
]
selected_page = st.sidebar.radio("Demo Modules", pages)
st.sidebar.caption("Only the active module is loaded to reduce cloud startup time.")

if selected_page == pages[0]:
    a, b, c, d = st.columns(4)
    a.metric("Users", f"{len(users):,}")
    b.metric("Products", f"{len(products):,}")
    c.metric("Interactions", f"{len(interactions):,}")
    d.metric("Purchases", f"{(interactions.event_type == 'purchase').sum():,}")
    left, right = st.columns(2)
    funnel = behavior_funnel(interactions)
    left.plotly_chart(
        px.funnel(funnel, x="count", y="event_type", title="User Behavior Funnel"), use_container_width=True
    )
    sensitivity = price_sensitivity(interactions, products)
    right.plotly_chart(
        px.bar(sensitivity, x="price_band", y="purchase_rate", title="Purchase Rate by Price Band"),
        use_container_width=True,
    )
    left2, right2 = st.columns(2)
    category = category_performance(interactions, products)
    category["category"] = category["category"].replace(CATEGORY_EN)
    left2.plotly_chart(
        px.bar(
            category, x="category", y="conversion_rate", color="refund_rate", title="Category Conversion and Refund Rate"
        ),
        use_container_width=True,
    )
    scene = scene_performance(interactions)
    scene["scene"] = scene["scene"].replace(SCENE_EN)
    right2.plotly_chart(
        px.bar(scene, x="scene", y="conversion_rate", title="Conversion Rate by Shopping Context"), use_container_width=True
    )

if selected_page == pages[1]:
    category_options = ["All"] + sorted(products["category"].unique().tolist())
    selected = st.selectbox(
        "Category", category_options, format_func=lambda value: CATEGORY_EN.get(value, value)
    )
    budget = st.slider("Maximum Budget", 50, 2000, 600, 50)
    result = popular_recommend(
        products, interactions, None if selected == "All" else selected, budget
    )
    st.dataframe(
        display_products(
            result[
                ["product_id", "title", "category", "brand", "price", "rating", "score", "reason"]
            ]
        ),
        use_container_width=True,
    )

if selected_page == pages[2]:
    st.subheader("Multi-Channel Retrieval with Reciprocal Rank Fusion")
    recall_left, recall_right = st.columns(2)
    user_id = recall_left.selectbox("User", users["user_id"].head(100).tolist())
    query = recall_right.text_input("Search Query", "commute minimalist")
    recall_category = recall_left.selectbox(
        "Retrieval Category",
        ["All"] + sorted(products["category"].unique().tolist()),
        format_func=lambda value: CATEGORY_EN.get(value, value),
    )
    recall_budget = recall_right.number_input(
        "Budget Limit", min_value=50, max_value=3000, value=800, step=50
    )
    hybrid = hybrid_recommend(
        products,
        interactions,
        user_id=user_id,
        query=query,
        category=None if recall_category == "All" else recall_category,
        budget_max=float(recall_budget),
        top_k=10,
    )
    st.dataframe(
        display_products(
            hybrid[
                [
                    "product_id",
                    "title",
                    "category",
                    "brand",
                    "price",
                    "score",
                    "evidence",
                    "reason",
                ]
            ]
        ),
        use_container_width=True,
    )
    st.caption(
        "Implemented retrieval channels include popularity, ItemCF, recent interests, co-basket, BM25, and TF-IDF vectors. "
        "Offline experiments determine which channels enter the default ensemble."
    )
if selected_page == pages[3]:
    st.subheader("LLM Agent: Intent Understanding, Tool Use, and Grounded Answers")
    m8_path = ARTIFACT_DIR / "m8_shopping_assistant_eval.json"
    if m8_path.exists():
        m8 = json.loads(m8_path.read_text(encoding="utf-8"))
        eval_columns = st.columns(3)
        eval_columns[0].metric("Regression Cases", m8["cases"])
        eval_columns[1].metric("Slot Accuracy", f"{m8['slot_accuracy']:.0%}")
        eval_columns[2].metric("Constraint Satisfaction", f"{m8['constraint_satisfaction_rate']:.0%}")
        st.caption("Metrics come from a hand-designed regression set with 6 requests and 36 slots. They measure functional regression, not open-domain generalization.")
    st.caption(
        "This demo calls the OpenAI Responses API. The model must first choose either clarification or catalog search. "
        "Recommended product IDs are validated against retrieved results, with automatic fallback to local rules on API failure."
    )
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    model_name = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    max_session_calls = 3
    if "llm_demo_calls" not in st.session_state:
        st.session_state.llm_demo_calls = 0
    remaining_calls = max(max_session_calls - st.session_state.llm_demo_calls, 0)
    status_columns = st.columns(2)
    status_columns[0].metric("Model", model_name)
    remaining_calls_metric = status_columns[1].empty()
    remaining_calls_metric.metric("Calls Remaining in This Session", remaining_calls)
    assistant_need = st.text_area(
        "Describe what you want to buy",
        "I need minimalist women's clothing for commuting under ¥500. Exclude the Xingye brand.",
        height=100,
    )
    assistant_user = st.selectbox(
        "Linked User (Optional)", ["Do not use history"] + users["user_id"].astype(str).head(100).tolist()
    )
    if api_key:
        st.success("The LLM service is securely configured on the server. No API key is required from visitors.")
    else:
        st.info("The LLM service is not configured. The deployer must set OPENAI_API_KEY in Streamlit Secrets.")
    if remaining_calls == 0:
        st.warning("The LLM demo limit for this browser session has been reached. Refresh the session to try again.")
    if st.button(
        "Run LLM Shopping Agent",
        type="primary",
        disabled=not bool(api_key) or remaining_calls == 0,
    ):
        st.session_state.llm_demo_calls += 1
        remaining_calls = max(max_session_calls - st.session_state.llm_demo_calls, 0)
        remaining_calls_metric.metric("Calls Remaining in This Session", remaining_calls)
        selected_user: object | None = None
        if assistant_user != "Do not use history":
            user_series = users.loc[users["user_id"].astype(str) == assistant_user, "user_id"]
            selected_user = user_series.iloc[0]
        with st.spinner("The model is planning tools and generating a grounded answer…"):
            agent = create_openai_agent(products, interactions, api_key, model=model_name)
            agent_result = agent.run(assistant_need, user_id=selected_user)
        status_columns = st.columns(3)
        status_columns[0].metric("Status", agent_result.status)
        status_columns[1].metric("Model", agent_result.model or "-")
        status_columns[2].metric("Recommendations", len(agent_result.recommendations))
        if agent_result.status == "needs_clarification":
            st.warning(agent_result.clarification_question)
        elif agent_result.status == "fallback":
            st.warning(agent_result.answer)
        else:
            st.success(agent_result.answer)
        if agent_result.recommendations:
            st.dataframe(
                pd.DataFrame([item.model_dump() for item in agent_result.recommendations]),
                use_container_width=True,
            )
        with st.expander("View Structured Tool Trace"):
            st.json(agent_result.tool_trace)
            if agent_result.error:
                st.error(agent_result.error)
if selected_page == pages[4]:
    st.subheader("Lightweight Visual Product Search")
    st.warning(
        "This module uses synthetic thumbnails and color, spatial, and edge features to validate the visual retrieval pipeline. "
        "It is not CLIP and does not represent semantic retrieval performance on real product photos."
    )
    m9_path = ARTIFACT_DIR / "m9_visual_search_eval.json"
    if m9_path.exists():
        m9 = json.loads(m9_path.read_text(encoding="utf-8"))
        visual_metrics = st.columns(3)
        visual_metrics[0].metric("Brightness-Shift Queries", m9["cases"])
        visual_metrics[1].metric("Top-1", f"{m9['top1_accuracy']:.0%}")
        visual_metrics[2].metric("Top-5", f"{m9['top5_accuracy']:.0%}")
        st.caption("Evaluation scope: synthetic thumbnails with 0.72–1.28x brightness shifts. This is not a real-image generalization score.")
    visual_products = products.head(100).copy()
    visual_index = cached_visual_index(visual_products)
    source_mode = st.radio("Query Image Source", ["Demo Product", "Upload Image"], horizontal=True)
    query_image = None
    if source_mode == "Demo Product":
        sample_id = st.selectbox("Select Demo Product", visual_products["product_id"].astype(str).tolist())
        sample_position = visual_index.product_ids.index(sample_id)
        query_image = Image.open(visual_index.image_paths[sample_position])
        st.image(query_image, caption=f"Query image: {sample_id} (synthetic demo asset)", width=180)
    else:
        uploaded = st.file_uploader("Upload a PNG/JPG Image", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            query_image = Image.open(uploaded)
            st.image(query_image, caption="Uploaded query image", width=180)
    if st.button("Search Visually Similar Products", disabled=query_image is None):
        assert query_image is not None
        visual_result = visual_search(query_image, visual_products, visual_index, top_k=5)
        result_columns = st.columns(5)
        for column, row in zip(result_columns, visual_result.itertuples(index=False), strict=True):
            column.image(row.image_path, use_container_width=True)
            column.markdown(f"**{TITLE_EN.get(row.title, row.title)}**")
            column.caption(
                f"{row.product_id} · {CATEGORY_EN.get(row.category, row.category)} · ¥{row.price:.2f}"
            )
            column.progress(
                max(0.0, min(1.0, row.visual_score)), text=f"Similarity {row.visual_score:.3f}"
            )
if selected_page == pages[5]:
    st.subheader("Offline Evaluation and Performance Metrics")
    eval_users = st.slider("Evaluation Users", 10, 100, 30, 10)
    if st.button("Run Leave-One-Out Evaluation"):
        with st.spinner("Running evaluation…"):
            metrics = evaluate_hybrid(products, interactions, k=10, max_users=eval_users)
        metric_columns = st.columns(4)
        metric_columns[0].metric("HitRate@10", f"{metrics['hit_rate_at_k']:.2%}")
        metric_columns[1].metric("NDCG@10", f"{metrics['ndcg_at_k']:.3f}")
        metric_columns[2].metric("Catalog Coverage", f"{metrics['catalog_coverage']:.2%}")
        metric_columns[3].metric("P95 Latency", f"{metrics['p95_latency_ms']:.1f} ms")
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
                title="Model Comparison on Real Data",
            ),
            use_container_width=True,
        )
        st.dataframe(comparison, use_container_width=True)
    if bad_case_path.exists():
        details = pd.read_csv(bad_case_path)
        bad_cases = details[details["hit"] == False]
        st.metric("Missed Bad Cases", f"{len(bad_cases)} / {len(details)}")
        st.dataframe(bad_cases.head(100), use_container_width=True)
    m5_path = ARTIFACT_DIR / "m5_weight_tuning.json"
    m5_bad_case_path = ARTIFACT_DIR / "m5_bad_cases.csv"
    if m5_path.exists():
        m5 = json.loads(m5_path.read_text(encoding="utf-8"))
        st.subheader("Weighted Fusion Holdout Results")
        weighted = m5["test_metrics"]
        weighted_columns = st.columns(4)
        weighted_columns[0].metric("HitRate@10", f"{weighted['hit_rate_at_k']:.2%}")
        weighted_columns[1].metric("NDCG@10", f"{weighted['ndcg_at_k']:.4f}")
        weighted_columns[2].metric("Coverage", f"{weighted['catalog_coverage']:.2%}")
        weighted_columns[3].metric("P95", f"{weighted['p95_latency_ms']:.1f} ms")
        st.caption(f"Weights selected on the validation set: {m5['best_weights']}")
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
            px.pie(distribution, names="type", values="count", title="Bad Case Type Distribution"),
            use_container_width=True,
        )
    m6_path = ARTIFACT_DIR / "m6_recall_expansion.json"
    if m6_path.exists():
        m6 = json.loads(m6_path.read_text(encoding="utf-8"))
        m6_results = pd.DataFrame(m6["results"])
        st.subheader("M6 Retrieval Channel Ablation")
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
            "P95 Latency",
            f"{best['p95_latency_ms']:.1f} ms",
            f"{best['p95_latency_ms'] - base['p95_latency_ms']:+.1f} ms",
            delta_color="inverse",
        )
        st.caption(
            f"Fixed holdout set: {m6['test_slice']}; best configuration: {m6['best_model']}. "
            "Co-basket retrieval did not improve this slice and was excluded from the default ensemble."
        )
        st.plotly_chart(
            px.bar(
                m6_results,
                x="model",
                y=["hit_rate_at_k", "candidate_recall"],
                barmode="group",
                title="Expanded Retrieval Channel Ablation",
            ),
            use_container_width=True,
        )
    m7_path = ARTIFACT_DIR / "m7_ranking_experiment.json"
    if m7_path.exists():
        m7 = json.loads(m7_path.read_text(encoding="utf-8"))
        m7_results = pd.DataFrame(m7["results"])
        baseline = m7_results.iloc[0]
        reranked = m7_results.iloc[1]
        accepted = reranked["hit_rate_at_k"] > baseline["hit_rate_at_k"]
        st.subheader("M7 Learning-to-Rank and Offline Release Gate")
        gate_columns = st.columns(4)
        gate_columns[0].metric("Training Rows", f"{m7['training']['training_rows']:,}")
        gate_columns[1].metric("Positive Rows", f"{m7['training']['positive_rows']}")
        gate_columns[2].metric(
            "Reranked HitRate@10",
            f"{reranked['hit_rate_at_k']:.2%}",
            f"{reranked['hit_rate_at_k'] - baseline['hit_rate_at_k']:+.2%}",
        )
        gate_columns[3].metric("Release Gate", "Passed" if accepted else "Rejected")
        if accepted:
            st.success("The reranker outperforms the current baseline and can proceed to staged validation.")
        else:
            st.warning(
                "The reranker does not beat the M6 baseline, so M6 remains the default model. "
                "The main limitation is that the candidate pool contains only 20 positive training examples."
            )
        st.caption(f"Ranking strategy: {m7['ranking_strategy']}; features: {', '.join(m7['features'])}")
        st.plotly_chart(
            px.bar(
                m7_results,
                x="model",
                y=["hit_rate_at_k", "ndcg_at_k", "catalog_coverage"],
                barmode="group",
                title="M6 vs. M7 on the Holdout Set",
            ),
            use_container_width=True,
        )
