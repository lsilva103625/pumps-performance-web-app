# app_web_final_v5.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# --- Cargar modelos (opcional)
try:
    model_eff = joblib.load('model_efficiency.pkl')
    model_pow = joblib.load('model_power.pkl')
except FileNotFoundError:
    model_eff = model_pow = None
    st.warning("⚠️ Modelos IA no encontrados. Usando interpolación física.")

# Datos reales extraídos del PDF GORMANN RUPP 95_140 PERFORMACE CURVES.pdf
DATA = {
    "S8A-E95": pd.DataFrame({
        'Q': [0, 500, 1000, 1500, 2000, 2500, 3000],
        'H': [140.0, 135.0, 125.0, 110.0, 90.0, 65.0, 30.0],
        'E': [0.0, 60.0, 75.0, 82.0, 80.0, 70.0, 50.0],
        'P': [20.0, 35.0, 50.0, 65.0, 80.0, 100.0, 120.0]
    }),
    "S12A-E140": pd.DataFrame({
        'Q': [0, 1000, 2000, 3000, 4000, 5000, 6000],
        'H': [130.0, 125.0, 115.0, 100.0, 80.0, 55.0, 20.0],
        'E': [0.0, 65.0, 78.0, 85.0, 83.0, 75.0, 55.0],
        'P': [25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 170.0]
    })
}

# Inicializar session_state solo si no existen las claves
if 'sp_gr' not in st.session_state:
    st.session_state.sp_gr = 1.0
if 'rpm' not in st.session_state:
    st.session_state.rpm = 1750.0
if 'visc' not in st.session_state:
    st.session_state.visc = 1.0
if 'temp' not in st.session_state:
    st.session_state.temp = 20.0

# Configuración
st.set_page_config(
    page_title="Simulador de Operación v5",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Simulador de Operación — Gorman-Rupp")
st.caption("Curvas predichas bajo condiciones no nominales")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    model = st.selectbox("Modelo", ["S8A-E95", "S12A-E140"])
    
    st.subheader("Parámetros IA")
    # Asignamos el valor del session_state al slider sin usar 'key'
    sp_gr = st.slider("Sp. Gravity", 0.8, 1.2, st.session_state.sp_gr, step=0.01)
    rpm   = st.slider("RPM", 1500.0, 1800.0, st.session_state.rpm, step=10.0)
    visc  = st.slider("Viscosity (cP)", 0.8, 2.0, st.session_state.visc, step=0.01)
    temp  = st.slider("Temp (°C)", 10.0, 50.0, st.session_state.temp, step=0.1)

    # Botón Reset: actualiza el session_state y recarga
    if st.button("🔄 Reset a valores de fábrica"):
        st.session_state.sp_gr = 1.0
        st.session_state.rpm = 1750.0
        st.session_state.visc = 1.0
        st.session_state.temp = 20.0
        st.rerun() # Recarga la app con los nuevos valores

# Actualizar session_state con los valores actuales de los sliders
# Esto asegura que si el usuario mueve los sliders, el estado se mantenga
st.session_state.sp_gr = sp_gr
st.session_state.rpm = rpm
st.session_state.visc = visc
st.session_state.temp = temp

# Cargar datos
df = DATA[model]

# Funciones de predicción física (si no hay modelo IA)
def predict_efficiency_curve(q_vals, e_nominal, sp_gr, rpm):
    """E_pred ≈ E_nominal * (Sp.Gr.)^0.1 * (RPM/1750)^0.2"""
    base_rpm = 1750.0
    factor = (sp_gr ** 0.1) * ((rpm / base_rpm) ** 0.2)
    return np.clip(e_nominal * factor, 0, 100)  # No >100%

def predict_power_curve(q_vals, p_nominal, sp_gr, rpm):
    """P_pred = P_nominal * Sp.Gr. * (RPM/1750)^3"""
    base_rpm = 1750.0
    factor = sp_gr * (rpm / base_rpm) ** 3
    return p_nominal * factor

# Calcular curvas predichas
e_pred_curve = predict_efficiency_curve(df['Q'], df['E'], sp_gr, rpm)
p_pred_curve = predict_power_curve(df['Q'], df['P'], sp_gr, rpm)

# Simulador de operación
st.subheader("🔍 Simulador de Operación")
q_val = st.slider("Capacidad (GPM)", float(df['Q'].min()), float(df['Q'].max()), float(df['Q'].mean()), step=10.0)

# Interpolación en puntos
h_val = np.interp(q_val, df['Q'], df['H'])
e_val = np.interp(q_val, df['Q'], df['E'])
p_val = np.interp(q_val, df['Q'], df['P'])

# Predicción puntual (IA o física)
if model_eff is not None:
    try:
        X = np.array([[q_val, h_val, sp_gr, rpm, visc, temp]])
        e_pred_point = float(model_eff.predict(X)[0])
        p_pred_point = float(model_pow.predict(X)[0])
    except Exception:
        e_pred_point, p_pred_point = e_val, p_val
else:
    e_pred_point, p_pred_point = e_val, p_val

# Mostrar métricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Q", f"{q_val:.0f} GPM")
col2.metric("H", f"{h_val:.1f} ft")
col3.metric("E", f"{e_pred_point:.1f} %")
col4.metric("P", f"{p_pred_point:.1f} kW")

# Gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Curvas nominales
ax.plot(df['Q'], df['H'], label='Altura Total (ft)', color='#1a5276', linewidth=2.2, marker='o', markersize=4)
ax2 = ax.twinx()
ax2.plot(df['Q'], df['E'], label='Eficiencia (%) [Nominal]', color='#c0392b', linewidth=2.2, marker='s', markersize=4)
ax3 = ax.twinx()
ax3.spines.right.set_position(("outward", 60))
ax3.plot(df['Q'], df['P'], label='Potencia (kW) [Nominal]', color='#27ae60', linewidth=2.2, marker='^', markersize=4)

# Curvas predichas
ax2.plot(df['Q'], e_pred_curve, '--', label='Eficiencia (%) [Predicción]', color='#c0392b', linewidth=2.0, zorder=5)
ax3.plot(df['Q'], p_pred_curve, '--', label='Potencia (kW) [Predicción]', color='#27ae60', linewidth=2.0, zorder=5)

# Ejes y grid
ax.set_xlabel('Capacidad (GPM)', fontsize=12)
ax.set_ylabel('Altura (ft)', color='#1a5276', fontsize=12)
ax2.set_ylabel('Eficiencia (%)', color='#c0392b', fontsize=12)
ax3.set_ylabel('Potencia (kW)', color='#27ae60', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.4)

# Ajustar límites Y
ax.set_ylim(0, df['H'].max() * 1.05)
ax2.set_ylim(0, max(100, e_pred_curve.max() * 1.1))
ax3.set_ylim(0, max(180, p_pred_curve.max() * 1.1))

# Título
fig.suptitle(f"Curva: {model} + Predicción IA", fontsize=14, fontweight='bold')

# Leyenda personalizada en el centro inferior del gráfico
legend_text = (
    "— Altura Total (ft)\n"
    "— Eficiencia (%) [Nominal]\n"
    "— Potencia (kW) [Nominal]\n"
    "··· Eficiencia (%) [Predicción]\n"
    "··· Potencia (kW) [Predicción]"
)
ax.text(0.5, 0.01, legend_text,
        transform=ax.transAxes,
        fontsize=8,
        ha='center',
        va='bottom',
        color='black',
        bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor='#d0d0d0', alpha=0.95))

# Punto dorado (simulado)
ax.plot([q_val], [h_val], 'o', color='gold', markersize=12, linewidth=2.5, markeredgecolor='black', zorder=10)

# Mostrar gráfico
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("💡 App desarrollada por Tecnolotls — Ingeniería Industrial Inteligente | Basado en curvas oficiales Gorman-Rupp")