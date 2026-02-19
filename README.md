# Digital Alchemy: ML for Reaction Yield Prediction
Machine learning for predicting Buchwald-Hartwig C-N coupling reaction yields. Compares one-hot encoding vs transformer-based approaches (ChemBERTa) on the Ahneman dataset (4312 reactions).

**Course Project**: Developed as part of an ML course, including chemistry sessions and practical exercises.

---

## Results
| Approach | RMSE | MAE | R² |
|----------|------|-----|-----|
| **One-Hot + Neural Network** | **7.8%** | **5.6%** | **0.9** |
| One-Hot + Gradient Boosting | 11.1% | 8.3% | 0.8 |
| ChemBERTa + Neural Network | 16.2% | 11.5% | 0.6 |

**Finding**: One-hot encoding significantly outperforms transformers for fixed-component datasets (46 unique molecules).

---

## Quick Start

**Requirements:** Python 3.11
```bash
# Create virtual environment with Python 3.11
python3.11 -m venv alchemy_env
source alchemy_env/bin/activate  # Linux/Mac
# alchemy_env\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt

# Run web app
python yield_predictor_Gradio.py
```

**Web Interface**: Select catalyst, aryl halide, base, and additive → Get instant yield prediction

**Note**: Python 3.11 is required for proper dependency resolution between TensorFlow and ord-schema.

---

## Repository
- `yield_prediction_ahneman.ipynb`: One-hot encoding approach
- `yield_prediction_Reaction_Smiles.ipynb`: ChemBERTa approach  
- `yield_predictor_Gradio.py`: Interactive web app
- `yield_model.keras`: Trained model
- `requirements.txt`: Python dependencies
- `Alchemy.pdf`: Full technical report

---

**Dataset**: Ahneman Buchwald-Hartwig from [Open Reaction Database](https://open-reaction-database.org)
