import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Retail Dashboard", layout="wide")

st.markdown("""
<style>
body {
    background-color: #020617;
}

.main {
    background-color: #020617;
    color: white;
}

h1, h2, h3, h4, h5, h6 {
    color: #e2e8f0 !important;
}

label, .stSelectbox label {
    color: #cbd5f5 !important;
}

.stSelectbox div {
    background-color: #1e293b !important;
    color: white !important;
}

[data-baseweb="select"] * {
    color: white !important;
}

.stMultiSelect div {
    background-color: #1e293b !important;
    color: white !important;
}

.stSidebar {
    background-color: #020617 !important;
}

</style>
""", unsafe_allow_html=True)

df = pd.read_csv("C:/Users/User/Downloads/Quarterly_Retail_Sales_Tax_Data_by_County_and_City.csv")
df.columns = [col.strip() for col in df.columns]
st.sidebar.header("🔍 Filters")

year = st.sidebar.selectbox("Year", sorted(df["Year"].unique()))
county = st.sidebar.selectbox("County", df["County"].unique())
quarter = st.sidebar.multiselect("Quarter", df["Quarter"].unique())

filtered = df[(df["Year"] == year) & (df["County"] == county)]
if quarter:
    filtered = filtered[filtered["Quarter"].isin(quarter)]


st.title("🚀 Retail Sales Intelligence Dashboard")


total_sales = filtered["Taxable Sales"].sum()
avg_sales = filtered["Taxable Sales"].mean()
total_tax = filtered["Computed Tax"].sum()
growth = filtered["Computed Tax"].pct_change().mean() * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Average Sales", f"${avg_sales:,.0f}")
col3.metric("Total Tax", f"${total_tax:,.0f}")
col4.metric("Growth Rate", f"{growth:.2f}%")

st.subheader("📈 Year-wise Sales Trend")

yearly = df.groupby("Year")["Taxable Sales"].sum().reset_index()

fig1 = px.line(yearly, x="Year", y="Taxable Sales")

fig1.update_layout(
    plot_bgcolor="#020617",
    paper_bgcolor="#020617",
    font=dict(color="white"),
    xaxis=dict(color="white"),
    yaxis=dict(color="white")
)

st.plotly_chart(fig1, use_container_width=True)


st.subheader("🏆 Top Counties")

top = df.groupby("County")["Taxable Sales"].sum().nlargest(10).reset_index()

fig2 = px.bar(top, x="County", y="Taxable Sales")

fig2.update_layout(
    plot_bgcolor="#020617",
    paper_bgcolor="#020617",
    font=dict(color="white"),
    xaxis=dict(color="white"),
    yaxis=dict(color="white")
)

st.plotly_chart(fig2, use_container_width=True)


st.subheader("🧩 Quarter Distribution")

q = df["Quarter"].value_counts().reset_index()
q.columns = ["Quarter", "Count"]

fig3 = px.pie(q, names="Quarter", values="Count")

fig3.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="black")
)

st.plotly_chart(fig3, use_container_width=True)


st.subheader("💰 Tax vs Sales (ML)")

fig4 = px.scatter(
    filtered,
    x="Taxable Sales",
    y="Computed Tax",
    trendline="ols"
)

fig4.update_layout(
    plot_bgcolor="#020617",
    paper_bgcolor="#020617",
    font=dict(color="white"),
    xaxis=dict(color="white"),
    yaxis=dict(color="white")
)

st.plotly_chart(fig4, use_container_width=True)

st.subheader("🤖 Sales Prediction")

X = df[["Taxable Sales"]]
y = df["Computed Tax"]

model = LinearRegression()
model.fit(X, y)

user_input = st.number_input("Enter Sales")

if st.button("Predict Tax"):
    prediction = model.predict([[user_input]])
    st.success(f"Predicted Tax: ${prediction[0]:,.2f}")


st.subheader("📋 Data Table")
st.dataframe(filtered)