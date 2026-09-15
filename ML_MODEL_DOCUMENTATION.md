# ML Model Documentation - Livestock Health Detector

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Training Pipeline (test2.py)](#training-pipeline-test2py)
4. [Prediction System (app.py)](#prediction-system-apppy)
5. [Feature Engineering](#feature-engineering)
6. [Model Artifacts](#model-artifacts)
7. [Why Each Component is Used](#why-each-component-is-used)

---

## Overview

This is a **Hierarchical Two-Stage Animal Disease Prediction System** that uses Machine Learning to predict diseases in livestock based on symptoms, vital signs, and clinical observations.

### Key Features
- **Animal-Specific Models**: Separate models trained for each animal type (Dog, Cat, Cow, Horse, Pig, Goat, Sheep, Rabbit)
- **Two-Stage Prediction**:
  - Stage 1: Predict the syndrome (Respiratory, GI, Multi-System, etc.)
  - Stage 2: Predict the specific disease based on the syndrome
- **Ensemble Learning**: Combines RandomForest, XGBoost, and LightGBM for better accuracy
- **Probability Calibration**: Uses CalibratedClassifierCV to provide reliable confidence scores
- **Class Imbalance Handling**: Uses SMOTE and undersampling to handle rare diseases

---

## Architecture

```
Input Features
      ↓
[Syndrome Classifier] ← Stage 1
      ↓
  Syndrome
      ↓
[Disease Classifier] ← Stage 2 (per syndrome)
      ↓
Disease Prediction + Confidence
```

### Why Hierarchical?
1. **Improved Accuracy**: Breaking the problem into two stages reduces complexity
2. **Better Generalization**: Syndromes are more generalizable than specific diseases
3. **Handles Data Sparsity**: Some diseases have few samples; grouping by syndrome helps
4. **Clinical Relevance**: Mirrors real veterinary diagnosis process

---

## Training Pipeline (test2.py)

### 1. **Data Loading and Preprocessing**

```python
df = pd.read_csv('cleaned_animal_disease_prediction.csv')
```

**What it does**: Loads the training dataset containing animal disease records

**Why**: This CSV contains historical data with symptoms, vital signs, and confirmed diagnoses

---

### 2. **Duration Parsing**

```python
def parse_duration_to_days(x):
    # Converts "5 days", "1 week", "2 weeks" to numeric days
```

**Why we use this**:
- Duration data comes in mixed formats (days/weeks/strings)
- ML models need numeric values
- Converts everything to consistent unit (days)
- Handles edge cases (NaN, already numeric, etc.)

---

### 3. **Temperature and Heart Rate Parsing**

```python
def parse_temperature(x):
    # Strips '°', 'C' and converts to float
```

**Why we use this**:
- Temperature data may include units (°C, C, degrees)
- Needs cleaning to extract numeric values
- Missing values filled with median (39.0°C for most animals)

---

### 4. **Binary Symptom Encoding**

```python
yesno_cols = ['Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing', ...]
for c in yesno_cols:
    df[c] = df[c].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
```

**Why we use this**:
- ML models work better with numeric data (0/1) than text (Yes/No)
- Binary encoding preserves the meaning while being computationally efficient
- Standardizes inconsistent inputs (Yes/yes/YES → 1)

---

### 5. **Syndrome Labeling**

```python
def syndrome_label(row):
    resp = row.get('Coughing', 0) + row.get('Labored_Breathing', 0) + ...
    gi = row.get('Vomiting', 0) + row.get('Diarrhea', 0) + ...
    # Choose highest scoring syndrome or 'Multi' if multiple
```

**Why we use this**:
- Creates Stage 1 target labels based on symptom patterns
- Groups similar diseases together (e.g., all respiratory diseases)
- Syndromes:
  - **Respiratory**: Coughing, labored breathing, nasal discharge
  - **GI (Gastrointestinal)**: Vomiting, diarrhea, appetite loss
  - **Dermatological**: Skin lesions
  - **Neurological**: Lameness
  - **Systemic**: Abnormal temperature or heart rate
  - **Multi**: Multiple systems affected

**Clinical Justification**: 
- Veterinarians diagnose by first identifying the affected body system
- This mimics real-world clinical reasoning

---

### 6. **Rare Label Merging**

```python
def merge_rare_labels(df, threshold=3):
    # Merges diseases with <3 samples into 'Other' category
```

**Why we use this**:
- Diseases with very few training samples lead to overfitting
- Model can't learn patterns from 1-2 examples
- Grouping rare diseases into "Other" prevents poor predictions
- Maintains model reliability and prevents false confidence

**Trade-off**: Lose specific diagnosis for rare diseases, but gain overall accuracy

---

### 7. **Label Encoding for Categories**

```python
label_encoders = {}
for c in ['Breed', 'Gender', 'Symptom_1', ...]:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])
    label_encoders[c] = le
```

**Why we use this**:
- Converts categorical text (e.g., "Labrador", "Male") to numbers (0, 1, 2, ...)
- ML models require numeric input
- LabelEncoder is saved so we can transform new data consistently during prediction

---

### 8. **Feature Columns**

```python
feature_cols = [
    'Breed', 'Age', 'Gender', 'Weight',
    'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4',
    'Duration_days', 'Body_Temperature', 'Heart_Rate',
    'Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing', ...
]
```

**Why these features**:
- **Breed**: Different breeds have different disease susceptibilities
- **Age**: Young and senior animals more vulnerable
- **Gender**: Some diseases are gender-specific
- **Weight**: Indicator of overall health and size
- **Symptoms 1-4**: Primary clinical observations
- **Duration**: Acute vs chronic conditions
- **Vital Signs**: Temperature and heart rate indicate severity
- **Binary Symptoms**: Detailed clinical picture

---

### 9. **Train/Calibration/Test Split**

```python
def safe_train_calib_test_split(X, y, test_size=0.15, calib_size=0.15):
    # Splits data into: 70% train, 15% calibration, 15% test
```

**Why we use this**:
- **Training Set (70%)**: Used to train the model
- **Calibration Set (15%)**: Used to calibrate probability predictions
- **Test Set (15%)**: Used to evaluate final performance

**Why stratified split?**
- Ensures each split has representative samples of all classes
- Prevents bias from having all samples of one disease in training

**Why calibration set?**
- Raw model probabilities are often poorly calibrated
- Calibration ensures confidence scores are reliable (e.g., 80% confidence means correct 80% of the time)

---

### 10. **Standard Scaling**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Why we use this**:
- Features have different ranges (Age: 0-20, Weight: 5-500, Temperature: 37-41)
- Without scaling, high-value features dominate the model
- StandardScaler: transforms to mean=0, std=1
- **Result**: All features contribute equally to predictions

**Formula**: `(value - mean) / standard_deviation`

---

### 11. **SMOTE (Synthetic Minority Over-sampling)**

```python
smote = SMOTE(random_state=42, k_neighbors=3)
X_res, y_res = smote.fit_resample(X_scaled, y_encoded)
```

**Why we use this**:
- **Problem**: Disease dataset is imbalanced (some diseases have 100 samples, others have 5)
- **Without SMOTE**: Model learns to always predict common diseases
- **With SMOTE**: Creates synthetic samples for rare diseases
- **How it works**: For each minority sample, creates new samples by interpolating between k-nearest neighbors

**Example**:
```
Before SMOTE:
- Mastitis: 100 samples
- Ketosis: 10 samples

After SMOTE:
- Mastitis: 100 samples
- Ketosis: 80 samples (70 synthetic)
```

---

### 12. **RandomUnderSampler**

```python
rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_res, y_res = rus.fit_resample(X_scaled, y_encoded)
```

**Why we use this**:
- Used BEFORE SMOTE
- Reduces majority class samples to prevent overwhelming minority classes
- **Combined Strategy**: Undersample majority + Oversample minority = Balanced dataset

---

### 13. **Syndrome Classifier (Stage 1)**

```python
rf_synd = RandomForestClassifier(
    n_estimators=300, 
    max_depth=15, 
    class_weight='balanced', 
    random_state=42
)
```

**Why RandomForest for syndromes**:
- **Robust**: Handles mixed data types well
- **Fast**: Syndromes are few (5-6 classes), so training is quick
- **Interpretable**: Can see which features are important
- **Parameters**:
  - `n_estimators=300`: Uses 300 decision trees (more trees = more accurate)
  - `max_depth=15`: Limits tree depth to prevent overfitting
  - `class_weight='balanced'`: Gives equal importance to all syndromes
  - `min_samples_split=2`: Minimum samples to split a node

---

### 14. **CalibratedClassifierCV**

```python
calib_synd = CalibratedClassifierCV(estimator=rf_synd, cv='prefit')
calib_synd.fit(X_calib_s_sc, y_calib_s)
```

**Why we use this**:
- **Problem**: RandomForest probabilities are often overconfident or underconfident
- **Solution**: Calibration adjusts probabilities to match actual accuracy
- **cv='prefit'**: Uses already-trained model, calibrates on separate calibration set

**What it does**:
```
Before calibration:
Model says 90% confidence, but actually correct 70% of the time

After calibration:
Model says 70% confidence, and is correct 70% of the time
```

**Why it matters**: Doctors/farmers need to trust the confidence scores for decision-making

---

### 15. **Disease Classifiers (Stage 2) - Ensemble**

Three models are trained per syndrome:

#### a) **RandomForest**

```python
RandomForestClassifier(n_estimators=300, max_depth=15, ...)
```

**Why**:
- **Strengths**: Robust, handles non-linear relationships, less prone to overfitting
- **Use case**: Good baseline, works well with limited data

#### b) **XGBoost**

```python
XGBClassifier(n_estimators=150, use_label_encoder=False, eval_metric='mlogloss')
```

**Why**:
- **Strengths**: State-of-art performance, handles missing values, fast training
- **Use case**: Often achieves highest accuracy on structured data
- **Gradient Boosting**: Learns from mistakes of previous trees

#### c) **LightGBM**

```python
LGBMClassifier(n_estimators=150, class_weight='balanced', verbosity=-1)
```

**Why**:
- **Strengths**: Very fast, memory efficient, handles large datasets
- **Use case**: Good for real-time predictions
- **Leaf-wise growth**: More efficient than level-wise growth

---

### 16. **Ensemble Prediction (Averaging)**

```python
prob_list = []
for model in models.values():
    prob_list.append(model.predict_proba(X_test))
avg_proba = np.mean(prob_list, axis=0)
```

**Why ensemble**:
- Each model has different strengths and weaknesses
- Averaging reduces variance and improves reliability
- **Result**: More stable and accurate predictions than any single model

**Analogy**: Like getting a second opinion from multiple doctors

---

### 17. **Top-K Accuracy**

```python
def top_k_accuracy(y_true, y_pred_topk, k=3):
    # Checks if true disease is in top 3 predictions
```

**Why we use this**:
- Top-1 (exact match) can be too strict in medical diagnosis
- Top-3 gives veterinarian a differential diagnosis (3 most likely diseases)
- More practical for real-world use

**Example**:
```
True disease: Pneumonia
Predictions: 
1. Pneumonia (40%)
2. Bronchitis (35%)
3. Pleurisy (15%)

Top-1 accuracy: ✓ Correct
Top-3 accuracy: ✓ Correct (Pneumonia is in top 3)
```

---

### 18. **Model Saving (Joblib)**

```python
joblib.dump({
    'classifier': synd_clf,
    'scaler': scaler_synd,
    'label_encoder': le_synd
}, 'syndrome_clf.joblib')
```

**Why we use this**:
- Saves trained models to disk
- Can load and use models without retraining
- **Joblib**: Optimized for large numpy arrays (better than pickle)
- **What's saved**:
  - Trained model weights
  - Scaler parameters (mean, std)
  - Label encoder mappings

---

## Prediction System (app.py)

### 1. **AnimalSpecificDiseasePredictor Class**

```python
class AnimalSpecificDiseasePredictor:
    def __init__(self):
        self.animal_models = {}
        self.animal_scalers = {}
        self.animal_encoders = {}
```

**Why we use this**:
- Encapsulates all prediction logic in one place
- Manages different models for different animals
- Provides clean interface for predictions

---

### 2. **Feature Preparation**

```python
def _prepare_input_features(self, breed, age, gender, ...):
    input_data = {
        'Breed': breed,
        'Age': age,
        ...
    }
```

**Why we use this**:
- Converts user input into model-ready format
- Applies same preprocessing as training
- Calculates derived features (explained below)

---

### 3. **Species-Specific Vital Sign Analysis**

```python
normal_ranges = {
    'Dog': {'temp': (38.0, 39.2), 'hr': (60, 160)},
    'Cat': {'temp': (38.1, 39.2), 'hr': (140, 220)},
    'Horse': {'temp': (37.2, 38.6), 'hr': (28, 44)},
    ...
}
```

**Why we use this**:
- Normal vital signs vary dramatically by species
- Dog HR: 60-160 BPM vs Cat HR: 140-220 BPM
- Detecting abnormalities requires species-specific thresholds
- **Creates features**:
  - `Temp_Abnormal`: -1 (low), 0 (normal), 1 (high)
  - `Fever_Severity`: How far from normal
  - `HR_Abnormal`: -1 (low), 0 (normal), 1 (high)
  - `HR_Severity`: Degree of abnormality

---

### 4. **Syndrome Scores**

```python
input_data['Respiratory_Syndrome'] = (
    input_data['Coughing'] * 3 + 
    input_data['Labored_Breathing'] * 4 +
    input_data['Nasal_Discharge'] * 2 + 
    input_data['Eye_Discharge'] * 1
)
```

**Why we use this**:
- Creates composite scores for each body system
- **Weights**: More severe symptoms have higher weights
  - Labored Breathing (4) more serious than Eye Discharge (1)
- **Result**: Single number representing system involvement
- Helps model recognize patterns across syndromes

**Other syndrome scores**:
- **GI_Syndrome**: Vomiting (4) + Diarrhea (3) + Appetite Loss (2)
- **Systemic_Syndrome**: Temperature abnormality + Appetite loss
- **Dermatological_Syndrome**: Skin lesions
- **Neurological_Syndrome**: Lameness

---

### 5. **Duration-Based Conditions**

```python
input_data['Acute_Condition'] = 1 if duration <= 3 else 0
input_data['Chronic_Condition'] = 1 if duration > 14 else 0
```

**Why we use this**:
- Distinguishes disease progression patterns
- **Acute** (<3 days): Sudden onset, often infectious
- **Chronic** (>14 days): Long-term, often metabolic or degenerative
- **Subacute** (3-14 days): In between
- Different diseases have different timelines

---

### 6. **Multi-System Disease Detection**

```python
system_count = sum([
    input_data['Respiratory_Syndrome'] > 2,
    input_data['GI_Syndrome'] > 2,
    input_data['Systemic_Syndrome'] > 2,
    ...
])
input_data['Multi_System_Disease'] = 1 if system_count >= 2 else 0
```

**Why we use this**:
- Some diseases affect multiple body systems (e.g., sepsis, distemper)
- Indicates disease severity
- Helps model recognize complex conditions
- Triggers more urgent veterinary recommendation

---

### 7. **Age and Size Factors**

```python
input_data['Young_Animal'] = 1 if age < 2 else 0
input_data['Senior_Animal'] = 1 if age > 8 else 0
input_data['Small_Animal'] = 1 if weight < 30 else 0
input_data['Large_Animal'] = 1 if weight > 200 else 0
```

**Why we use this**:
- **Age-related vulnerability**:
  - Young animals: Weaker immune systems, more prone to infections
  - Senior animals: Prone to degenerative diseases
- **Size-related patterns**:
  - Small animals: Different disease profiles, different dosing
  - Large animals: Different biomechanics, different stresses

---

### 8. **Two-Stage Prediction Process**

```python
# STAGE 1: Predict syndrome
synd_proba = synd_clf.predict_proba(X_scaled)[0]
synd_idx = np.argmax(synd_proba)
syndrome_label = le_synd.inverse_transform([synd_idx])[0]

# STAGE 2: Predict disease using syndrome-specific model
disease_models = artifacts['disease_models'][syndrome_label]
avg_proba = np.mean([model.predict_proba(X) for model in models], axis=0)
predicted_disease = le_disease.inverse_transform([np.argmax(avg_proba)])[0]
```

**Flow**:
1. Input features → Syndrome Classifier → "Respiratory Syndrome" (80% confidence)
2. Load Respiratory-specific disease models
3. Run ensemble of 3 models
4. Average their probabilities
5. Select disease with highest probability

---

### 9. **Confidence Adjustment**

```python
adjusted_confidence = 0.7 * disease_confidence + 0.3 * syndrome_confidence
```

**Why we use this**:
- Final confidence depends on BOTH stages
- Weighted average (70% disease, 30% syndrome)
- Prevents overconfidence when syndrome prediction is uncertain
- **Example**:
  ```
  Syndrome confidence: 60%
  Disease confidence: 90%
  Adjusted: 0.7 * 0.9 + 0.3 * 0.6 = 0.81 (81%)
  ```

---

### 10. **Top-3 Predictions**

```python
top_indices = np.argsort(avg_proba)[::-1][:3]
top_predictions = []
for idx in top_indices:
    disease = le_disease.inverse_transform([idx])[0]
    probability = avg_proba[idx]
    top_predictions.append({'disease': disease, 'probability': probability})
```

**Why we use this**:
- Provides differential diagnosis
- Helps veterinarians consider multiple possibilities
- More clinically useful than single prediction
- Shows uncertainty in model's decision

---

## Feature Engineering

### Engineered Features Summary

| Feature | Purpose | Example |
|---------|---------|---------|
| `Temp_Abnormal` | Categorize temperature | -1 (low), 0 (normal), 1 (high) |
| `Fever_Severity` | Magnitude of temperature deviation | 0.5 = half degree above normal |
| `HR_Abnormal` | Categorize heart rate | -1 (low), 0 (normal), 1 (high) |
| `Respiratory_Syndrome` | Composite respiratory score | Sum of weighted respiratory symptoms |
| `GI_Syndrome` | Composite GI score | Sum of weighted GI symptoms |
| `Acute_Condition` | Quick onset indicator | 1 if ≤3 days, 0 otherwise |
| `Chronic_Condition` | Long-term disease indicator | 1 if >14 days, 0 otherwise |
| `Multi_System_Disease` | Severity indicator | 1 if ≥2 systems affected |
| `Young_Animal` | Vulnerability factor | 1 if <2 years old |
| `Senior_Animal` | Age-related risk | 1 if >8 years old |

**Why feature engineering**:
- Raw features alone don't capture relationships
- Derived features encode domain knowledge
- Helps model learn faster and more accurately
- Reduces need for massive training data

---

## Model Artifacts

### Saved Files Structure

```
./models/
├── Dog/
│   ├── animal_artifacts.joblib        # Main metadata
│   ├── syndrome_clf.joblib            # Stage 1 classifier
│   ├── disease_models_Respiratory.joblib  # Stage 2 for Respiratory
│   ├── disease_models_GI.joblib       # Stage 2 for GI
│   └── disease_models_Multi.joblib    # Stage 2 for Multi-system
├── Cat/
│   ├── ... (same structure)
└── Cow/
    └── ... (same structure)
```

### Contents of Each Artifact

#### 1. `animal_artifacts.joblib`
```python
{
    'disease_models': {
        'Respiratory': {...},
        'GI': {...},
        'Multi': {...}
    },
    'syndrome_encoder': LabelEncoder(),
    'syndrome_scaler': StandardScaler(),
    'feature_columns': [...],
    'label_encoders_cat': {...},
    'model_metrics': {
        'accuracy': 0.85,
        'precision': 0.83,
        ...
    }
}
```

#### 2. `syndrome_clf.joblib`
```python
{
    'classifier': CalibratedClassifierCV(RandomForestClassifier),
    'scaler': StandardScaler(),
    'label_encoder': LabelEncoder(['Respiratory', 'GI', 'Multi', ...])
}
```

#### 3. `disease_models_<syndrome>.joblib`
```python
{
    'models': {
        'rf': CalibratedClassifierCV(RandomForest),
        'xgb': CalibratedClassifierCV(XGBoost),
        'lgb': CalibratedClassifierCV(LightGBM)
    },
    'label_encoder': LabelEncoder(['Pneumonia', 'Bronchitis', ...]),
    'scaler': StandardScaler()
}
```

---

## Why Each Component is Used

### Summary Table

| Component | Why Used | Alternative | Trade-off |
|-----------|----------|-------------|-----------|
| **Hierarchical (2-stage)** | Reduces complexity, mirrors clinical process | Single-stage classifier | More complex, but more accurate |
| **SMOTE** | Handles class imbalance | Class weights only | Better for rare diseases |
| **RandomForest** | Robust, interpretable | Neural networks | Less accurate, but more reliable with limited data |
| **XGBoost** | State-of-art accuracy | RandomForest only | Slower, but worth it for accuracy |
| **LightGBM** | Fast inference | XGBoost only | Slightly less accurate, but much faster |
| **Ensemble Averaging** | Reduces variance | Single best model | More computation, but more reliable |
| **Calibration** | Reliable confidence scores | Raw probabilities | Essential for medical decisions |
| **StandardScaler** | Equal feature importance | No scaling | Prevents feature dominance |
| **Species-specific ranges** | Accurate vital sign assessment | Generic thresholds | More complex, but medically sound |
| **Syndrome scores** | Encodes clinical patterns | Raw symptoms only | Helps model learn faster |
| **Top-K accuracy** | Practical for medical use | Top-1 only | More forgiving, more useful |

---

## Performance Metrics

### What's Measured

1. **Accuracy**: Percentage of correct predictions
   - `(True Positives + True Negatives) / Total Samples`

2. **Precision**: Of all positive predictions, how many were correct?
   - `True Positives / (True Positives + False Positives)`
   - **Why important**: Reduces false alarms

3. **Recall**: Of all actual positives, how many did we catch?
   - `True Positives / (True Positives + False Negatives)`
   - **Why important**: Don't miss serious diseases

4. **F1-Score**: Harmonic mean of precision and recall
   - `2 * (Precision * Recall) / (Precision + Recall)`
   - **Why important**: Balanced measure

5. **Top-3 Accuracy**: Is true disease in top 3 predictions?
   - **Why important**: Practical for differential diagnosis

### Example Output

```
Dog Model Metrics:
  Accuracy: 0.847 (84.7%)
  Precision: 0.831
  Recall: 0.842
  F1-Score: 0.836
  Samples: 1250
  Diseases: 18
  
  Syndrome test accuracy: 0.923
  Disease Top-1=0.847, Top-3=0.956
```

---

## Conclusion

This ML system combines:
- **Domain Knowledge**: Veterinary science encoded in features
- **Advanced ML**: Ensemble methods, calibration, imbalance handling
- **Practical Design**: Two-stage hierarchy, differential diagnosis
- **Reliability**: Calibrated probabilities, extensive validation

The result is a clinically useful tool that provides:
1. Primary disease prediction with confidence
2. Top 3 differential diagnoses
3. Syndrome classification
4. Vital signs analysis
5. Severity assessment

**Clinical Impact**: Helps farmers and veterinarians make faster, more informed decisions about livestock health.
