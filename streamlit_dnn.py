import streamlit as st
import tokens_dnn

st.write("""
# Opini Masyarakat di Twitter
""")

input_text = st.text_input('Teks Judul Analisis Twitter', '')

if st.button("Detection"):
    label_dnn = tokens_dnn.predict_model_dnn(input_text)
    sentiment = "Positif" if label_dnn[0][0] > 0.5 else "Negatif"
    score = label_dnn[0][0] if sentiment == "Positif" else 1 - label_dnn[0][0]
    
    if sentiment == "Positif":
        st.markdown("""
            <style>
            div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.1);
            padding: 1% 1% 1% 1%;
            border-radius: 2px;
            color: rgb(13, 252, 13);
            overflow-wrap: break-word;
            }

            /* breakline for metric text         */
            div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
            overflow-wrap: break-word;
            white-space: break-spaces;
            color: green;
            font-size: 200%;
            }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.1);
            padding: 1% 1% 1% 1%;
            border-radius: 2px;
            color: rgb(255, 0, 0);
            overflow-wrap: break-word;
            }

            /* breakline for metric text         */
            div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
            overflow-wrap: break-word;
            white-space: break-spaces;
            color: red;
            font-size: 200%;
            }
            </style>
            """, unsafe_allow_html=True)

    st.metric("Sentimen", sentiment, f"{score * 100:.2f}%")
