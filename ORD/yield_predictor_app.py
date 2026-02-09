import gradio as gr
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras

# Component SMILES mappings (from your notebook)
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
        'P2-Et base': 'CCN=P(N=P(N(C)C)(N(C)C)N(C)C)(N(C)C)N(C)C',
        'TMG': 'CN(C)C(=NC(C)(C)C)N(C)C',
        'DBU': 'CN1CCCN2CCCN=C12',
    },
    'additive': {
        'DMSO': 'CS(=O)C',
        '3-phenylisoxazole': 'c1ccc(-c2ccno2)cc1',
        'ethyl 3-methylisoxazole-5-carboxylate': 'CCOC(=O)c1cc(C)no1',
        'ethyl 5-methylisoxazole-3-carboxylate': 'CCOC(=O)c1cc(C)on1',
        '5-phenylisoxazole': 'c1ccc(-c2cnoc2)cc1',
        '3-phenylisoxazole (alt)': 'c1ccc(-c2ccon2)cc1',
        '3-methylisoxazole': 'Cc1ccon1',
        '3-phenyl-1,2,4-oxadiazole': 'c1ccc(-c2ncno2)cc1',
        '5-methylisoxazole': 'Cc1ccno1',
        'benzisoxazole': 'c1ccc2nocc2c1',
        '3,5-dimethylisoxazole': 'Cc1cc(C)on1',
        'methyl isoxazole-5-carboxylate': 'COC(=O)c1ccno1',
        'ethyl isoxazole-3-carboxylate': 'CCOC(=O)c1ccon1',
        'ethyl 5-methylisoxazole-4-carboxylate': 'CCOC(=O)c1cnoc1C',
        'ethyl isoxazole-4-carboxylate': 'CCOC(=O)c1cnoc1',
        'benzoxazole': 'c1ccc2oncc2c1',
        'ethyl 5-methoxyisoxazole-3-carboxylate': 'CCOC(=O)c1cc(OC)no1',
        '5-methyl-3-phenylisoxazole': 'Cc1cc(-c2ccccc2)on1',
        '3-dibenzylaminoisoxazole': 'c1ccc(CN(Cc2ccccc2)c2ccon2)cc1',
        'methyl 3-(2-furyl)isoxazole-5-carboxylate': 'COC(=O)c1cc(-c2ccco2)on1',
        '3-(2,6-difluorophenyl)isoxazole': 'Fc1cccc(F)c1-c1ccno1',
        '3-dibenzylaminoisoxazole (alt)': 'c1ccc(CN(Cc2ccccc2)c2ccno2)cc1',
        '5-methyl-3-(1H-pyrrol-1-yl)isoxazole': 'Cc1cc(-n2cccc2)no1',
        'methyl 3-(2-thienyl)isoxazole-5-carboxylate': 'COC(=O)c1cc(-c2cccs2)on1',
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
    X = create_input_vector(catalyst_smiles, aryl_halide_smiles, base_smiles, additive_smiles)
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
            classification = "✅ Success (≥50%)"
            color = "green"
        else:
            classification = "❌ Failure (<50%)"
            color = "red"
        
        result = f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #f0f0f0;">
            <h2 style="color: {color}; margin-bottom: 10px;">Predicted Yield: {yield_percent:.1f}%</h2>
            <h3 style="color: {color};">{classification}</h3>
            <hr>
            <p><strong>Reaction Components:</strong></p>
            <ul>
                <li><strong>Catalyst:</strong> {catalyst}</li>
                <li><strong>Aryl Halide:</strong> {aryl_halide}</li>
                <li><strong>Base:</strong> {base}</li>
                <li><strong>Additive:</strong> {additive}</li>
                <li><strong>Amine:</strong> p-toluidine</li>
            </ul>
        </div>
        """
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Buchwald-Hartwig Yield Predictor", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🧪 Buchwald-Hartwig C-N Coupling Yield Predictor
        
        Predict the yield of Pd-catalyzed C-N cross-coupling reactions using a trained neural network.
        
        **Select your reaction components:**
        """
    )
    
    with gr.Row():
        with gr.Column():
            catalyst_dropdown = gr.Dropdown(
                choices=list(COMPONENTS['catalyst'].keys()),
                label="Catalyst",
                value=list(COMPONENTS['catalyst'].keys())[0]
            )
            aryl_halide_dropdown = gr.Dropdown(
                choices=list(COMPONENTS['aryl_halide'].keys()),
                label="Aryl Halide",
                value=list(COMPONENTS['aryl_halide'].keys())[0]
            )
        
        with gr.Column():
            base_dropdown = gr.Dropdown(
                choices=list(COMPONENTS['base'].keys()),
                label="Base",
                value=list(COMPONENTS['base'].keys())[0]
            )
            additive_dropdown = gr.Dropdown(
                choices=list(COMPONENTS['additive'].keys()),
                label="Additive",
                value=list(COMPONENTS['additive'].keys())[0]
            )
    
    predict_btn = gr.Button("🔬 Predict Yield", variant="primary", size="lg")
    
    output = gr.HTML(label="Prediction Result")
    
    predict_btn.click(
        fn=predict_yield,
        inputs=[catalyst_dropdown, aryl_halide_dropdown, base_dropdown, additive_dropdown],
        outputs=output
    )
    
    gr.Markdown(
        """
        ---
        **Model:** Neural Network (64→32 neurons with dropout)  
        **Dataset:** 4312 Buchwald-Hartwig reactions from Ahneman et al.  
        **Performance:** RMSE = 7.0%, MAE = 4.9%, R² = 0.93
        """
    )

if __name__ == "__main__":
    app.launch(share=True)
