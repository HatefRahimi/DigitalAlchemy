import gradio as gr
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras

COMPONENTS = {
    'catalyst': {
        'Catalyst 1 (Pd-XPhos)': 'CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)(C2CCCCC2)->[Pd]2(OS(=O)(=O)C(F)(F)F)<-Nc3ccccc3-c3ccccc32)c(C(C)C)c1',
        'Catalyst 2 (Pd-tBuXPhos)': 'CC(C)c1cc(C(C)C)c(-c2ccccc2P(C(C)(C)C)(C(C)(C)C)->[Pd]2(OS(=O)(=O)C(F)(F)F)<-Nc3ccccc3-c3ccccc32)c(C(C)C)c1',
        'Catalyst 3 (Pd-BrettPhos)': 'COc1ccc(OC)c(P(C(C)(C)C)(C(C)(C)C)->[Pd]2(OS(=O)(=O)C(F)(F)F)<-Nc3ccccc3-c3ccccc32)c1-c1c(C(C)C)cc(C(C)C)cc1C(C)C',
        'Catalyst 4 (Pd-AdBrettPhos)': 'COc1ccc(OC)c(P(C23CC4CC(CC(C4)C2)C3)(C23CC4CC(CC(C4)C2)C3)->[Pd]2(OS(=O)(=O)C(F)(F)F)<-Nc3ccccc3-c3ccccc32)c1-c1c(C(C)C)cc(C(C)C)cc1C(C)C',
    },
    'aryl_halide': {
        '4-CF3-phenyl chloride': 'FC(F)(F)c1ccc(Cl)cc1',
        '4-CF3-phenyl bromide': 'FC(F)(F)c1ccc(Br)cc1',
        '4-CF3-phenyl iodide': 'FC(F)(F)c1ccc(I)cc1',
        '4-OMe-phenyl chloride': 'COc1ccc(Cl)cc1',
        '4-OMe-phenyl bromide': 'COc1ccc(Br)cc1',
        '4-OMe-phenyl iodide': 'COc1ccc(I)cc1',
        '4-Et-phenyl chloride': 'CCc1ccc(Cl)cc1',
        '4-Et-phenyl bromide': 'CCc1ccc(Br)cc1',
        '4-Et-phenyl iodide': 'CCc1ccc(I)cc1',
        '2-chloropyridine': 'Clc1ccccn1',
        '2-bromopyridine': 'Brc1ccccn1',
        '2-iodopyridine': 'Ic1ccccn1',
        '3-chloropyridine': 'Clc1cccnc1',
        '3-bromopyridine': 'Brc1cccnc1',
        '3-iodopyridine': 'Ic1cccnc1',
    },
    'base': {
        'P2Et': 'CCN=P(N=P(N(C)C)(N(C)C)N(C)C)(N(C)C)N(C)C',
        'BTMG': 'CN(C)C(=NC(C)(C)C)N(C)C',
        'MTBD': 'CN1CCCN2CCCN=C12',
    },
    'additive': {
        'DMSO': 'CS(=O)C',
        '5-phenyl-1,2-oxazole': 'c1ccc(-c2ccno2)cc1',
        'ethyl 3-methyl-1,2-oxazole-5-carboxylate': 'CCOC(=O)c1cc(C)no1',
        'ethyl 5-methyl-1,2-oxazole-3-carboxylate': 'CCOC(=O)c1cc(C)on1',
        '4-phenyl-1,2-oxazole': 'c1ccc(-c2cnoc2)cc1',
        '3-phenyl-1,2-oxazole': 'c1ccc(-c2ccon2)cc1',
        '3-methyl-1,2-oxazole': 'Cc1ccon1',
        '5-phenyl-1,2,4-oxadiazole': 'c1ccc(-c2ncno2)cc1',
        '5-methyl-1,2-oxazole': 'Cc1ccno1',
        '2,1-benzoxazole': 'c1ccc2nocc2c1',
        '3,5-dimethyl-1,2-oxazole': 'Cc1cc(C)on1',
        'methyl 1,2-oxazole-5-carboxylate': 'COC(=O)c1ccno1',
        'ethyl 1,2-oxazole-3-carboxylate': 'CCOC(=O)c1ccon1',
        'ethyl 5-methyl-1,2-oxazole-4-carboxylate': 'CCOC(=O)c1cnoc1C',
        'ethyl 1,2-oxazole-4-carboxylate': 'CCOC(=O)c1cnoc1',
        '1,2-benzoxazole': 'c1ccc2oncc2c1',
        'ethyl 3-methoxy-1,2-oxazole-5-carboxylate': 'CCOC(=O)c1cc(OC)no1',
        '3-methyl-5-phenyl-1,2-oxazole': 'Cc1cc(-c2ccccc2)on1',
        '3-(N,N-dibenzylamino)isoxazole': 'c1ccc(CN(Cc2ccccc2)c2ccon2)cc1',
        'methyl 5-(furan-2-yl)-1,2-oxazole-3-carboxylate': 'COC(=O)c1cc(-c2ccco2)on1',
        '5-(2,6-difluorophenyl)-1,2-oxazole': 'Fc1cccc(F)c1-c1ccno1',
        '5-(N,N-dibenzylamino)isoxazole': 'c1ccc(CN(Cc2ccccc2)c2ccno2)cc1',
        '5-methyl-3-pyrrol-1-yl-1,2-oxazole': 'Cc1cc(-n2cccc2)no1',
        'methyl 5-thiophen-2-yl-1,2-oxazole-3-carboxylate': 'COC(=O)c1cc(-c2cccs2)on1',
    }
}

# Reverse mapping: SMILES -> feature name
SMILES_TO_FEATURE = {}
for component, options in COMPONENTS.items():
    for i, (label, smiles) in enumerate(options.items(), 1):
        SMILES_TO_FEATURE[smiles] = f"{component}_{i}"

# Create feature order (same as training)
FEATURE_COLUMNS = []
for comp in ['catalyst', 'aryl_halide', 'base', 'additive', 'toluidine']:
    if comp == 'toluidine':
        FEATURE_COLUMNS.append('toluidine_1')
    else:
        n_unique = len(COMPONENTS.get(comp, {}))
        FEATURE_COLUMNS.extend([f"{comp}_{i}" for i in range(1, n_unique + 1)])


def create_input_vector(catalyst_smiles, aryl_halide_smiles, base_smiles, additive_smiles):
    """Convert user selections to one-hot encoded feature vector"""
    vector = np.zeros(len(FEATURE_COLUMNS))

    # Map SMILES to feature names and set to 1
    for smiles in [catalyst_smiles, aryl_halide_smiles, base_smiles, additive_smiles]:
        if smiles in SMILES_TO_FEATURE:
            feature_name = SMILES_TO_FEATURE[smiles]
            if feature_name in FEATURE_COLUMNS:
                idx = FEATURE_COLUMNS.index(feature_name)
                vector[idx] = 1

    # Always set toluidine to 1 (only one variant)
    if 'toluidine_1' in FEATURE_COLUMNS:
        idx = FEATURE_COLUMNS.index('toluidine_1')
        vector[idx] = 1

    return vector


def predict_yield(catalyst, aryl_halide, base, additive):
    """Make prediction using trained model"""

    # Get SMILES from selections
    catalyst_smiles = COMPONENTS['catalyst'][catalyst]
    aryl_halide_smiles = COMPONENTS['aryl_halide'][aryl_halide]
    base_smiles = COMPONENTS['base'][base]
    additive_smiles = COMPONENTS['additive'][additive]

    # Create input vector
    X = create_input_vector(
        catalyst_smiles, aryl_halide_smiles, base_smiles, additive_smiles)
    X = X.reshape(1, -1).astype(np.float32)

    # Load model and predict
    try:
        model = keras.models.load_model('yield_model.keras')
        prediction = model.predict(X, verbose=0)[0][0]
        yield_percent = prediction * 100

        # Clamp to 0-100%
        yield_percent = max(0, min(100, yield_percent))

        # Classification
        if yield_percent >= 50:
            classification = "✅ Success"
            classification_detail = "Expected yield ≥50%"
            bg_color = "#d4edda"
            border_color = "#28a745"
            text_color = "#155724"
        else:
            classification = "❌ Failure"
            classification_detail = "Expected yield <50%"
            bg_color = "#f8d7da"
            border_color = "#dc3545"
            text_color = "#721c24"

        result = f"""
        <div style="
            padding: 30px;
            border-radius: 15px;
            background-color: {bg_color};
            border: 3px solid {border_color};
            margin-top: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
            <div style="text-align: center; margin-bottom: 25px;">
                <h1 style="color: {text_color}; margin: 0; font-size: 3em; font-weight: 700;">
                    {yield_percent:.1f}%
                </h1>
                <h2 style="color: {text_color}; margin: 10px 0 5px 0; font-size: 1.8em; font-weight: 600;">
                    {classification}
                </h2>
                <p style="color: {text_color}; margin: 0; font-size: 1.2em; opacity: 0.8;">
                    {classification_detail}
                </p>
            </div>
            
            <hr style="border: none; border-top: 2px solid {border_color}; margin: 25px 0; opacity: 0.3;">
            
            <div style="background-color: rgba(255,255,255,0.5); padding: 20px; border-radius: 10px;">
                <h3 style="color: {text_color}; margin-top: 0; font-size: 1.4em; font-weight: 600;">
                    Reaction Components
                </h3>
                <table style="width: 100%; border-collapse: collapse; font-size: 1.1em;">
                    <tr style="border-bottom: 1px solid rgba(0,0,0,0.1);">
                        <td style="padding: 10px; font-weight: 600; color: {text_color}; width: 30%;">⚛️ Catalyst:</td>
                        <td style="padding: 10px; color: {text_color};">{catalyst}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(0,0,0,0.1);">
                        <td style="padding: 10px; font-weight: 600; color: {text_color};">🔬 Aryl Halide:</td>
                        <td style="padding: 10px; color: {text_color};">{aryl_halide}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(0,0,0,0.1);">
                        <td style="padding: 10px; font-weight: 600; color: {text_color};">⚗️ Base:</td>
                        <td style="padding: 10px; color: {text_color};">{base}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(0,0,0,0.1);">
                        <td style="padding: 10px; font-weight: 600; color: {text_color};">💧 Additive:</td>
                        <td style="padding: 10px; color: {text_color};">{additive}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; font-weight: 600; color: {text_color};">🧪 Amine:</td>
                        <td style="padding: 10px; color: {text_color};">p-toluidine</td>
                    </tr>
                </table>
            </div>
        </div>
        """
        return result

    except Exception as e:
        return f"""
        <div style="padding: 20px; background-color: #f8d7da; border: 2px solid #dc3545; border-radius: 10px; color: #721c24;">
            <h3>❌ Error</h3>
            <p style="font-size: 1.1em;">{str(e)}</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Make sure 'yield_model.keras' is in the same directory as this script.</p>
        </div>
        """


# Custom CSS for better appearance
custom_css = """
#title {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 0.5em;
    color: #2c3e50;
}
#subtitle {
    text-align: center;
    font-size: 1.2em;
    color: #7f8c8d;
    margin-bottom: 2em;
}
.dropdown-label {
    font-size: 1.1em !important;
    font-weight: 600 !important;
    color: #34495e !important;
}
#predict-btn {
    font-size: 1.3em !important;
    padding: 15px !important;
    font-weight: bold !important;
    margin-top: 1em !important;
    margin-bottom: 1em !important;
}
#footer {
    text-align: center;
    margin-top: 2em;
    padding: 1.5em;
    background-color: #f8f9fa;
    border-radius: 10px;
    font-size: 1em;
    color: #495057;
}
"""

# Create Gradio interface
with gr.Blocks(title="Buchwald-Hartwig Yield Predictor", css=custom_css, theme=gr.themes.Default()) as app:
    gr.HTML(
        """
        <div id="title">🧪 Buchwald-Hartwig Yield Predictor</div>
        <div id="subtitle">Predict C-N coupling reaction yields using machine learning</div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            catalyst_dropdown = gr.Dropdown(
                choices=list(COMPONENTS['catalyst'].keys()),
                label="Catalyst",
                value=list(COMPONENTS['catalyst'].keys())[0],
                elem_classes="dropdown-label"
            )
            aryl_halide_dropdown = gr.Dropdown(
                choices=list(COMPONENTS['aryl_halide'].keys()),
                label="Aryl Halide",
                value=list(COMPONENTS['aryl_halide'].keys())[0],
                elem_classes="dropdown-label"
            )

        with gr.Column(scale=1):
            base_dropdown = gr.Dropdown(
                choices=list(COMPONENTS['base'].keys()),
                label="Base",
                value=list(COMPONENTS['base'].keys())[0],
                elem_classes="dropdown-label"
            )
            additive_dropdown = gr.Dropdown(
                choices=list(COMPONENTS['additive'].keys()),
                label="Additive",
                value=list(COMPONENTS['additive'].keys())[0],
                elem_classes="dropdown-label"
            )

    predict_btn = gr.Button(
        "🔬 Predict Yield", variant="primary", size="lg", elem_id="predict-btn")

    output = gr.HTML()

    predict_btn.click(
        fn=predict_yield,
        inputs=[catalyst_dropdown, aryl_halide_dropdown,
                base_dropdown, additive_dropdown],
        outputs=output
    )

    gr.HTML(
        """
        <div id="footer">
            <strong>Model:</strong> Neural Network <br>
            <strong>Dataset:</strong> 4312 Buchwald-Hartwig reactions from Ahneman.<br>
            <strong>Performance:</strong> RMSE = 7.0% | MAE = 4.9% | R² = 0.93
        </div>
        """
    )

if __name__ == "__main__":
    app.launch(share=True)
