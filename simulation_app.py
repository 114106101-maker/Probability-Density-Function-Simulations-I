
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================
# 頁面設定
# ============================================================
st.set_page_config(
    page_title="📊 常態分布電腦模擬",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main .block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
}
.card {
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    background: white;
    margin-bottom: 18px;
}
.kpi {
    padding: 18px;
    border-radius: 16px;
    background: #f8f9fa;
    border: 1px solid #e5e7eb;
    text-align: center;
}
.kpi-title {
    font-size: 14px;
    color: #6b7280;
}
.kpi-value {
    font-size: 25px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 常態分布電腦模擬實驗")
st.caption("驗證 68–95–99.7 法則，以及常態分布的百分位數與臨界值")

# ============================================================
# 側邊欄
# ============================================================
with st.sidebar:
    st.header("⚙️ 模擬設定")

    experiment = st.radio(
        "選擇實驗",
        [
            "① 68–95–99.7 法則",
            "② 百分位數與臨界值"
        ]
    )

    n = st.selectbox(
        "模擬樣本數 n",
        [100, 1000, 10000],
        index=2
    )

    seed = st.number_input(
        "隨機種子",
        min_value=0,
        max_value=999999,
        value=42,
        step=1
    )

    run = st.button(
        "🚀 開始模擬",
        type="primary",
        use_container_width=True
    )

# ============================================================
# 第一題
# ============================================================
if experiment.startswith("①"):

    st.header("第一題：驗證常態分布的 68–95–99.7 法則")

    mu = 100
    sigma = 15

    st.markdown("""
    <div class="card">
    <b>實驗設定</b><br>
    X ~ N(100, 15²)<br>
    理論平均數 μ = 100<br>
    理論標準差 σ = 15
    </div>
    """, unsafe_allow_html=True)

    rng = np.random.default_rng(seed)

    if run or "data1" not in st.session_state:
        st.session_state.data1 = rng.normal(mu, sigma, n)

    data = st.session_state.data1

    # --------------------------------------------------------
    # (a)
    # --------------------------------------------------------
    st.subheader("🅐 樣本平均數與樣本標準差")

    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'<div class="kpi"><div class="kpi-title">理論平均數 μ</div>'
            f'<div class="kpi-value">100</div></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="kpi"><div class="kpi-title">模擬平均數</div>'
            f'<div class="kpi-value">{sample_mean:.4f}</div></div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="kpi"><div class="kpi-title">理論標準差 σ</div>'
            f'<div class="kpi-value">15</div></div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f'<div class="kpi"><div class="kpi-title">模擬標準差</div>'
            f'<div class="kpi-value">{sample_std:.4f}</div></div>',
            unsafe_allow_html=True
        )

    result_a = pd.DataFrame({
        "項目": ["平均數", "標準差"],
        "理論值": [mu, sigma],
        "模擬值": [sample_mean, sample_std],
        "差異": [sample_mean - mu, sample_std - sigma]
    })

    st.dataframe(
        result_a.style.format({
            "理論值": "{:.4f}",
            "模擬值": "{:.4f}",
            "差異": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # (b)(c)
    # --------------------------------------------------------
    st.subheader("🅑🅒 區間比例與 68–95–99.7 法則比較")

    intervals = [
        ("85 ≤ X ≤ 115", 85, 115, 0.6826),
        ("70 ≤ X ≤ 130", 70, 130, 0.9545),
        ("55 ≤ X ≤ 145", 55, 145, 0.9974)
    ]

    rows = []

    for label, lower, upper, theory in intervals:
        simulated = np.mean((data >= lower) & (data <= upper))
        rows.append({
            "區間": label,
            "模擬比例": simulated,
            "理論比例": theory,
            "差異": simulated - theory
        })

    result_bc = pd.DataFrame(rows)

    st.dataframe(
        result_bc.style.format({
            "模擬比例": "{:.4%}",
            "理論比例": "{:.4%}",
            "差異": "{:.4%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # 圖表
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 600)
    pdf = norm.pdf(x, mu, sigma)

    ax.hist(
        data,
        bins=40,
        density=True,
        alpha=0.6,
        edgecolor="black",
        label="模擬資料"
    )

    ax.plot(
        x,
        pdf,
        linewidth=2,
        label="理論常態密度"
    )

    ax.axvline(mu, linestyle="--", linewidth=1.5, label="μ = 100")
    ax.axvline(mu-sigma, linestyle=":", linewidth=1)
    ax.axvline(mu+sigma, linestyle=":", linewidth=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Density")
    ax.set_title(f"N(100, 15²)：n = {n:,}")
    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------------
    # (d)
    # --------------------------------------------------------
    st.subheader("🅓 n = 100 與 n = 10,000 比較")

    rng_compare = np.random.default_rng(seed)
    data100 = rng_compare.normal(mu, sigma, 100)
    data10000 = rng_compare.normal(mu, sigma, 10000)

    compare_rows = []

    for name, values in [
        ("n = 100", data100),
        ("n = 10,000", data10000)
    ]:
        p1 = np.mean((values >= 85) & (values <= 115))
        p2 = np.mean((values >= 70) & (values <= 130))
        p3 = np.mean((values >= 55) & (values <= 145))

        compare_rows.append({
            "樣本數": name,
            "平均數": np.mean(values),
            "樣本標準差": np.std(values, ddof=1),
            "±1σ 比例": p1,
            "±2σ 比例": p2,
            "±3σ 比例": p3
        })

    compare_df = pd.DataFrame(compare_rows)

    st.dataframe(
        compare_df.style.format({
            "平均數": "{:.4f}",
            "樣本標準差": "{:.4f}",
            "±1σ 比例": "{:.2%}",
            "±2σ 比例": "{:.2%}",
            "±3σ 比例": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "結論：樣本數越大，模擬結果通常越穩定，也越接近理論值。"
    )

    # --------------------------------------------------------
    # 額外：資料預覽
    # --------------------------------------------------------
    with st.expander("🔍 查看部分模擬資料"):
        preview = pd.DataFrame({
            "模擬編號": np.arange(1, min(101, n + 1)),
            "X": data[:100]
        })
        st.dataframe(
            preview.style.format({"X": "{:.4f}"}),
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# 第二題
# ============================================================
else:

    st.header("第二題：常態分布的百分位數與臨界值")

    mu = 130
    sigma = 20

    st.markdown("""
    <div class="card">
    <b>實驗設定</b><br>
    X ~ N(130, 20²)<br>
    理論平均數 μ = 130<br>
    理論標準差 σ = 20
    </div>
    """, unsafe_allow_html=True)

    rng = np.random.default_rng(seed)

    if run or "data2" not in st.session_state:
        st.session_state.data2 = rng.normal(mu, sigma, n)

    data = st.session_state.data2

    # --------------------------------------------------------
    # (a)
    # --------------------------------------------------------
    st.subheader("🅐 理論機率 P(X > 160)")

    z = (160 - mu) / sigma
    theory_p = 1 - norm.cdf(z)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("標準化 Z", f"{z:.4f}")

    with c2:
        st.metric("P(X > 160)", f"{theory_p:.6f}")

    with c3:
        st.metric("百分比", f"{theory_p:.4%}")

    st.latex(
        r"Z=\frac{160-130}{20}=1.5"
    )

    st.latex(
        r"P(X>160)=P(Z>1.5)=1-0.9332=0.0668"
    )

    # --------------------------------------------------------
    # (b)
    # --------------------------------------------------------
    st.subheader("🅑 模擬資料中 X > 160 的比例")

    simulation_p = np.mean(data > 160)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("理論值", f"{theory_p:.4%}")

    with c2:
        st.metric("模擬值", f"{simulation_p:.4%}")

    with c3:
        st.metric(
            "差異",
            f"{simulation_p - theory_p:+.4%}"
        )

    # --------------------------------------------------------
    # (c)
    # --------------------------------------------------------
    st.subheader("🅒 最高 5% 的理論臨界值")

    z95 = norm.ppf(0.95)
    c_theory = mu + z95 * sigma

    st.latex(
        r"P(X>c)=0.05 \Rightarrow P(X\leq c)=0.95"
    )

    st.latex(
        rf"z_{{0.95}}={z95:.6f}"
    )

    st.latex(
        rf"c=130+({z95:.6f})(20)={c_theory:.6f}"
    )

    st.success(
        f"理論臨界值：c = {c_theory:.4f}"
    )

    # --------------------------------------------------------
    # (d)
    # --------------------------------------------------------
    st.subheader("🅓 模擬資料的第 95 百分位數")

    c_simulation = np.percentile(data, 95)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("理論臨界值", f"{c_theory:.4f}")

    with c2:
        st.metric("模擬第 95 百分位數", f"{c_simulation:.4f}")

    with c3:
        st.metric(
            "差異",
            f"{c_simulation - c_theory:+.4f}"
        )

    # 分布圖
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 600)
    pdf = norm.pdf(x, mu, sigma)

    ax.hist(
        data,
        bins=40,
        density=True,
        alpha=0.6,
        edgecolor="black",
        label="模擬資料"
    )

    ax.plot(
        x,
        pdf,
        linewidth=2,
        label="理論常態密度"
    )

    ax.axvline(
        160,
        linestyle="--",
        linewidth=2,
        label="X = 160"
    )

    ax.axvline(
        c_theory,
        linestyle=":",
        linewidth=2,
        label=f"理論 c = {c_theory:.2f}"
    )

    ax.set_xlabel("收縮壓 X")
    ax.set_ylabel("Density")
    ax.set_title(f"N(130, 20²)：n = {n:,}")
    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------------
    # (e)
    # --------------------------------------------------------
    st.subheader("🅔 重複 100 次，每次 n = 1,000")

    if st.button("🔄 執行 100 次百分位數模擬"):
        rng100 = np.random.default_rng(seed)

        percentiles = []

        progress = st.progress(0)

        for i in range(100):
            sample = rng100.normal(mu, sigma, 1000)
            percentiles.append(np.percentile(sample, 95))
            progress.progress((i + 1) / 100)

        progress.empty()

        percentiles = np.array(percentiles)

        st.session_state.percentiles = percentiles

    if "percentiles" in st.session_state:

        percentiles = st.session_state.percentiles

        result_e = pd.DataFrame({
            "統計量": [
                "理論第95百分位數",
                "100次估計平均",
                "估計值標準差",
                "最小值",
                "最大值"
            ],
            "數值": [
                c_theory,
                np.mean(percentiles),
                np.std(percentiles, ddof=1),
                np.min(percentiles),
                np.max(percentiles)
            ]
        })

        st.dataframe(
            result_e.style.format({"數值": "{:.4f}"}),
            use_container_width=True,
            hide_index=True
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(
            percentiles,
            bins=15,
            edgecolor="black",
            alpha=0.7
        )

        ax.axvline(
            c_theory,
            linestyle="--",
            linewidth=2,
            label=f"理論值 = {c_theory:.2f}"
        )

        ax.set_xlabel("第 95 百分位數估計值")
        ax.set_ylabel("次數")
        ax.set_title("100 次第 95 百分位數估計值的分布")
        ax.legend()
        ax.grid(alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "可以看到 100 個第 95 百分位數估計值會在理論值附近波動。"
        )

        with st.expander("🔍 查看 100 次估計結果"):
            percentile_df = pd.DataFrame({
                "第幾次": np.arange(1, 101),
                "第95百分位數": percentiles
            })

            st.dataframe(
                percentile_df.style.format({
                    "第95百分位數": "{:.4f}"
                }),
                use_container_width=True,
                hide_index=True
            )

# ============================================================
# 頁尾
# ============================================================
st.divider()

st.caption(
    "📚 常態分布模擬實驗｜使用 NumPy 產生隨機資料、SciPy 計算理論值、"
    "Matplotlib 繪製圖形、Streamlit 建立互動介面"
)
