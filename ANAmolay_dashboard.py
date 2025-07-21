if keyword:
    df = load_data(keyword)
    if not df.empty:
        st.write("### Raw Data", df)

        model = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = model.fit_predict(df[[keyword]])
        df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
        anomalies = df[df['anomaly'] == 1]

        st.write("### Anomalies", anomalies[['date', keyword]])

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['date'], df[keyword], label="Search Volume", color='blue')
        ax.scatter(anomalies['date'], anomalies[keyword], color='red', label="Anomalies")
        ax.set_title(f"Search Trend for '{keyword}' with Anomalies")
        ax.legend()
        st.pyplot(fig)

streamlit cache clear
streamlit run anomaly_dashboard.py

st.write("App started successfully!")

