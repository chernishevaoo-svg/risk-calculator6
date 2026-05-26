import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Калькулятор риска", page_icon="🏥", layout="wide")

# --- 2. ЗАГРУЗКА МОДЕЛИ И КОНФИГУРАЦИИ ---
@st.cache_resource
def load_data():
    model = joblib.load('model_pipeline.pkl')
    config = joblib.load('model_config.pkl')
    return model, config

model, config = load_data()

# --- 3. ИНТЕРФЕЙС ВВОДА ДАННЫХ ---
st.title("🏥 Оценка риска досуточной летальности")
st.markdown("Введите показатели пациента для расчета вероятности неблагоприятного исхода.")
st.markdown("---")

cols = st.columns(len(config['features']))
input_values = {}

for i, feature_name in enumerate(config['features']):
    with cols[i]:
        input_values[feature_name] = st.number_input(
            f"**{feature_name}**",
            value=0.0,
            step=0.1,
            format="%.2f"
        )

st.markdown("---")

# --- 4. РАСЧЕТ И ВЫВОД РЕЗУЛЬТАТОВ ---
if st.button("Рассчитать риск", type="primary", use_container_width=True):

    df_input = pd.DataFrame([input_values])

    for feat in config['inverted_features']:
        if feat in df_input.columns:
            max_val = config['inversion_values'][feat]
            df_input[feat] = max_val - df_input[feat]

    probability = model.predict_proba(df_input)[0, 1]
    prob_percent = probability * 100

    if probability < config['prob_low']:
        risk_group = "🟢 НИЗКИЙ РИСК"
        bg_color = "#d4edda"
        text_color = "#155724"
        recommendation = "Пациент может быть отнесен к группе рутинного наблюдения."
    elif probability >= config['prob_high']:
        risk_group = "🔴 ВЫСОКИЙ РИСК"
        bg_color = "#f8d7da"
        text_color = "#721c24"
        recommendation = "Рекомендуется интенсивное наблюдение и раннее агрессивное лечение."
    else:
        risk_group = "🟡 УМЕРЕННЫЙ РИСК"
        bg_color = "#fff3cd"
        text_color = "#856404"
        recommendation = "Необходимо усиленное внимание, мониторинг динамики показателей."

    st.markdown("### Результат оценки:")
    st.markdown(f"""
    <div style="background-color: {bg_color}; color: {text_color}; padding: 20px;
                border-radius: 10px; border: 1px solid {text_color};">
        <h2 style="text-align: center; margin: 0; color: {text_color};">{risk_group}</h2>
        <h3 style="text-align: center; margin: 10px 0 0 0; color: {text_color};">
            Вероятность летальности: <b>{prob_percent:.1f}%</b>
        </h3>
    </div>
    """, unsafe_allow_html=True)

    st.info(f"**Рекомендация:** {recommendation}")

    with st.expander("ℹ️ Подробнее о порогах стратификации"):
        st.write(f"Граница низкого риска: вероятность < **{config['prob_low']:.2f}**")
        st.write(f"Граница высокого риска: вероятность ≥ **{config['prob_high']:.2f}**")

    st.caption("⚠️ Внимание: данный калькулятор является вспомогательным инструментом научного исследования и не заменяет клинического суждения врача.")
