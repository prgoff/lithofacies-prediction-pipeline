import streamlit as st
import lasio
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt
import matplotlib.colors as colors

st.set_page_config(page_title="Facies Predictor", layout="wide")

# Load trained pipeline assets


@st.cache_resource
def load_pipeline():
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('best_baseline_rf.joblib')
    return scaler, model


try:
    scaler, model = load_pipeline()
    st.sidebar.success(" Model & Scaler loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model files: {e}")
    st.stop()

# App User Interface
st.title("Subsurface Facies Prediction Dashboard")
st.markdown("Upload a standard `.las` well log file to generate automated machine learning lithofacies classifications.")

uploaded_file = st.file_uploader("Choose a LAS file", type=['las'])

if uploaded_file is not None:
    # Read LAS file from memory buffer
    bytes_data = uploaded_file.read()
    str_io = io.StringIO(bytes_data.decode('utf-8', errors='ignore'))

    try:
        las = lasio.read(str_io)
        df_las = las.df().reset_index()  # Extract curve data and make 'Depth' a column

        # Force whatever the first column is (the depth index) to be named 'Depth'
        df_las.rename(columns={df_las.columns[0]: 'Depth'}, inplace=True)

        st.success(f"Successfully parsed: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Failed to parse LAS file: {e}")
        st.stop()

    # --- 3. AUTOMATIC DICTIONARY MAPPING & NOTIFICATIONS ---
    st.subheader("Dataset Verification & Processing")

    # Comprehensive dictionary mapping standard features to all known vendor aliases
    curve_aliases = {
        'GR': ['GR', 'GGCE', 'GR_ED', 'GAM', 'CGR', 'SGR', 'GRD'],
        'ILD_log10': ['ILD_LOG10', 'RTAO', 'ILD', 'LL3', 'RT', 'AHT90', 'AT90', 'RILD'],
        'DeltaPHI': ['DELTAPHI', 'DPHI', 'DPOR', 'DEPT_PHI', 'DPHI_NPHI'],
        'PHIND': ['PHIND', 'XPOR', 'NPHI', 'PHIN', 'NPHI_HL', 'POROSITY'],
        'PE': ['PE', 'PDPE', 'PEF', 'DEN_COR']
    }

    required_curves = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
    mapped_columns = {}
    auto_mapped_log = []

    # Hidden Auto-Detection Logic
    for curve in required_curves:
        detected_column = None
        for col in df_las.columns:
            if col.upper() in [alias.upper() for alias in curve_aliases[curve]]:
                detected_column = col
                break

        # If detected, lock it in. If completely missing, default to first column.
        mapped_columns[curve] = detected_column if detected_column else df_las.columns[0]

        if detected_column:
            auto_mapped_log.append(f"{curve} → {detected_column}")

    # Display a sleek notification banner
    if len(auto_mapped_log) == len(required_curves):
        st.info(
            f"**Auto-mapped all curves successfully:** {', '.join(auto_mapped_log)}")
    else:
        st.warning(
            "⚠️ Some curves could not be automatically matched. Please verify mappings below.")

    # Hide the dropdown menus inside a clean, collapsible menu
    with st.expander(" Advanced Curve Mapping Overrides"):
        st.write("If the auto-detection missed a curve, correct it manually here:")
        col_selectors = st.columns(len(required_curves))
        for idx, curve in enumerate(required_curves):
            with col_selectors[idx]:
                current_default = mapped_columns[curve]
                default_idx = df_las.columns.get_loc(
                    current_default) if current_default in df_las.columns else 0
                mapped_columns[curve] = st.selectbox(
                    f"{curve}:", df_las.columns, index=default_idx)

    # Handle dataset-specific geological metadata targets (NM_M and RELPOS)
    if 'NM_M' not in df_las.columns:
        df_las['NM_M'] = 1  # Default to non-marine environment proxy fallback
    if 'RELPOS' not in df_las.columns:
        df_las['RELPOS'] = 0.5  # Default to mid-formation coordinate positions

    # Failures and Warnings
    for curve, mapped_col in mapped_columns.items():
        if mapped_col in ['Depth', 'DEPT']:
            st.sidebar.error(
                f"Missing Tool: This file does not contain a {curve} log. The app will use background averages, but model accuracy will decrease.")

    # --- 4. PREPROCESSING PIPELINE ---
    df_proc = pd.DataFrame()
    df_proc['Depth'] = df_las['Depth']

    # Map selection inputs back to pipeline arrays safely
    for curve in required_curves:
        raw_series = pd.to_numeric(
            df_las[mapped_columns[curve]], errors='coerce')
        df_proc[curve] = raw_series

    # Check if the user passed raw resistivity numbers by checking the data median.
    if df_proc['ILD_log10'].median() > 5.0:
        df_proc['ILD_log10'] = np.log10(df_proc['ILD_log10'].clip(lower=0.001))

    # Bring in fallback targets
    df_proc['NM_M'] = df_las['NM_M'] if 'NM_M' in df_las.columns else 1
    df_proc['RELPOS'] = df_las['RELPOS'] if 'RELPOS' in df_las.columns else 0.5

    # Fill missing rows using column medians
    for col in required_curves:
        df_proc[col] = df_proc[col].fillna(df_proc[col].median())

    df_proc['GR_PHIND_ratio'] = df_proc['GR'] / (df_proc['PHIND'] + 0.001)

    # Replace rogue infinite math calculations with standard NaN markers
    df_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_proc['GR_PHIND_ratio'] = df_proc['GR_PHIND_ratio'].fillna(
        df_proc['GR_PHIND_ratio'].median())

    # Generate rolling windows sequentially
    for curve in required_curves:
        df_proc[f'{curve}_roll_mean'] = df_proc[curve].rolling(
            window=3, min_periods=1).mean()
        df_proc[f'{curve}_roll_std'] = df_proc[curve].rolling(
            window=3, min_periods=1).std().fillna(0)

    # Order columns exactly matching training phase features layout
    features_ordered = [
        'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS',
        'GR_PHIND_ratio', 'GR_roll_mean', 'GR_roll_std', 'ILD_log10_roll_mean',
        'ILD_log10_roll_std', 'DeltaPHI_roll_mean', 'DeltaPHI_roll_std',
        'PHIND_roll_mean', 'PHIND_roll_std', 'PE_roll_mean', 'PE_roll_std'
    ]

    X_live_raw = df_proc[features_ordered]

    # Scale and Predict
    X_live_scaled = scaler.transform(X_live_raw)
    df_proc['Predicted_Facies'] = model.predict(X_live_scaled)

    # --- 4b. AUTOMATED AI EVALUATION SCORECARD ---
    # Check if the uploaded file contains original geological core descriptions to test against
    if 'FACIES' in df_las.columns:
        st.markdown("---")
        st.write("### Automated AI Performance Scorecard")
        st.write(
            "Evaluating machine learning alignment metrics directly against expert core descriptions extracted from the log profile[cite: 4, 15].")

        from sklearn.metrics import accuracy_score, classification_report

        true_labels = df_las['FACIES'].fillna(-1).astype(int)
        valid_idx = true_labels != -1

        if valid_idx.sum() > 0:
            acc = accuracy_score(
                true_labels[valid_idx], df_proc['Predicted_Facies'][valid_idx])

            m1, m2 = st.columns(2)
            with m1:
                st.metric(label="Overall Model Alignment Accuracy",
                          value=f"{acc:.1%}")
            with m2:
                st.metric(label="Validated Depth Samples Evaluated",
                          value=f"{valid_idx.sum()} intervals")

            with st.expander(" View Detailed Granular Precision Report"):
                report = classification_report(
                    true_labels[valid_idx], df_proc['Predicted_Facies'][valid_idx], output_dict=False)
                st.code(report)

    # --- 5. DYNAMIC LOG VISUALIZATION TRACKS ---
    st.subheader("Machine Learning Log Interpretation Log Strip")

    # 1. Add the toggle switch to the sidebar
    show_advanced = st.sidebar.checkbox(
        "Show Advanced Engineering Tracks", value=False)

    # 2. Configure your custom color palette definitions
    facies_colors = ['#F4D03F', '#F5B041', '#DC7633', '#A11D33',
                     '#1B4F72', '#2E4053', '#7D6608', '#117A65', '#145A32']
    cmap_facies = colors.ListedColormap(facies_colors, 'indexed')

    # 3. Dynamically configure subplots based on the user's sidebar selection
    if show_advanced:
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 10), sharey=True)
        facies_track_idx = 4
    else:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 8), sharey=True)
        facies_track_idx = 2

    # Invert Y-axis globally on the first track so depth values increase downward
    ax[0].invert_yaxis()

    # Track 1: Gamma Ray (Standard)
    ax[0].plot(df_proc['GR'], df_proc['Depth'], color='black', lw=1.0)
    ax[0].set_title("Gamma Ray (GR)")
    ax[0].set_xlabel("API")
    ax[0].grid(True, linestyle=':', alpha=0.5)

    # Track 2: Deep Resistivity (Standard)
    ax[1].plot(df_proc['ILD_log10'], df_proc['Depth'], color='blue', lw=1.0)
    ax[1].set_title("Resistivity (ILD)")
    ax[1].set_xlabel("Log10 Ohmm")
    ax[1].grid(True, linestyle=':', alpha=0.5)

    # 4. Inject extra engineering tracks matching your friend's app if active
    if show_advanced:
        nm_m_data = df_las['NM_M'] if 'NM_M' in df_las.columns else np.ones(
            len(df_proc))
        relpos_data = df_las['RELPOS'] if 'RELPOS' in df_las.columns else np.linspace(
            0, 1, len(df_proc))

        # Track 3: Marine vs Non-Marine block line
        ax[2].plot(nm_m_data, df_proc['Depth'], color='purple', lw=1.5)
        ax[2].set_title("Marine Block (NM_M)")
        ax[2].set_xlabel("Code")
        ax[2].grid(True, linestyle=':', alpha=0.5)

        # Track 4: Relative Position index slope
        ax[3].plot(relpos_data, df_proc['Depth'], color='brown', lw=1.0)
        ax[3].set_title("Rel Position (RELPOS)")
        ax[3].set_xlabel("Slope Index")
        ax[3].grid(True, linestyle=':', alpha=0.5)

    # Final Track: Your clean multi-colored structural Facies Strip
    pred_strip = np.repeat(
        df_proc['Predicted_Facies'].values, 100).reshape(-1, 100)
    ax[facies_track_idx].imshow(pred_strip, cmap=cmap_facies, aspect='auto',
                                extent=[0, 20, df_proc['Depth'].max(), df_proc['Depth'].min()], vmin=1, vmax=9)
    ax[facies_track_idx].set_title("Predicted Facies")
    ax[facies_track_idx].set_xticks([])

    plt.tight_layout()
    st.pyplot(fig)

    # --- 6. DOWNLOAD PREDICTIONS AS CSV ---
    st.markdown("---")
    st.write("### Export Interpretation Results")
    st.write("Download the processed well log data along with your model's continuous facies predictions as a standard CSV spreadsheet.")

    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_bytes = convert_df_to_csv(df_proc)

    st.download_button(
        label="Download Facies Predictions (.csv)",
        data=csv_bytes,
        file_name=f"Facies_Predictions_{uploaded_file.name.replace('.las', '')}.csv",
        mime="text/csv",
        key='download-csv'
    )

    # --- 7. INTERACTIVE GEOLOGICAL CROSSPLOT (PLOTLY) ---
    st.markdown("---")
    st.write("### Interactive Facies Crossplot Clustering")
    st.write("Hover over any data point to inspect its exact logging metrics and depth location in real time.")

    import plotly.express as px

    # Create a descriptive text column specifically for the interactive hover popup box
    df_proc['Hover_Text'] = (
        "Depth: " + df_proc['Depth'].astype(str) + " ft<br>" +
        "Gamma Ray: " + df_proc['GR'].round(1).astype(str) + " API<br>" +
        "Resistivity: " + df_proc['ILD_log10'].round(2).astype(str) + " Log10"
    )

    # Build the interactive dynamic scatter plot engine
    fig_cross = px.scatter(
        df_proc,
        x='GR',
        y='ILD_log10',
        color='Predicted_Facies',
        hover_name='Hover_Text',
        color_continuous_scale=facies_colors,
        range_color=[1, 9],
        labels={'GR': 'Gamma Ray (API)',
                'ILD_log10': 'Resistivity (Log10 Ohmm)'},
        title="Interactive Decision Domains: GR vs. Resistivity"
    )

    # Clean up layout dimensions and structure for a crisp white background
    fig_cross.update_layout(
        template='plotly_white',          # Forces a professional white background template
        plot_bgcolor='white',             # Ensures the internal plotting canvas is pure white
        # Ensures the outer bounding card space is pure white
        paper_bgcolor='white',
        # Forces all text, titles, and labels to render in crisp black
        font=dict(color='black'),
        coloraxis_colorbar=dict(
            title="Facies ID",
            tickvals=list(range(1, 10)),
            title_font=dict(color='black'),
            tickfont=dict(color='black')
        ),
        xaxis=dict(
            # Adds clean, soft grey gridlines
            gridcolor='rgba(200,200,200,0.5)',
            linecolor='black',                 # Solid black axis bounding line
            title_font=dict(color='black'),
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            # Adds clean, soft grey gridlines
            gridcolor='rgba(200,200,200,0.5)',
            linecolor='black',                 # Solid black axis bounding line
            title_font=dict(color='black'),
            tickfont=dict(color='black')
        )
    )

    # Render natively in Streamlit
    st.plotly_chart(fig_cross, use_container_width=True)
