# An Efficient Transfer Learning with Prompt Learning for Brain Disorders Diagnosis

*(Accepted by **IEEE Journal of Biomedical and Health Informatics (JBHI)**, Special Issue on **Advancing Precision Medicine: Multi-Disciplinary Research and Emerging Strategies in Biomedical Informatics**, 2025)*  

---

## Data Preprocessing

The time-series fMRI data is divided using a **sliding-window mechanism**, followed by the calculation of the **Pearson Correlation Coefficient (PCC)**.  
For each segment, the PCC matrix is vectorized by taking the **upper-triangular part**, resulting in **6,670** connectivity features per sample.

For the **AD (Alzheimer’s Disease)** dataset, we additionally performed **data augmentation** to address the limited-sample issue and improve model generalization.  

- **Corresponding code:**  
  - `data_process.py`  
  - `data_process_dimension_transducer.py`  
  - `AD_data_process.py` *(for data augmentation in AD)*

---

## Experimental Setup

We design two **cross-disorder transfer learning** settings to comprehensively evaluate the **generalization** and **adaptability** of the proposed framework:

1. **BD ↔ MDD**: Bipolar Disorder and Major Depressive Disorder  
   - Both belong to *affective disorders*, sharing overlapping symptoms but distinct neural patterns.  
   - This setting evaluates **cross-affective transfer** performance.

2. **ASD ↔ AD**: Autism Spectrum Disorder and Alzheimer’s Disease  
   - Both belong to *neurodevelopmental and neurodegenerative disorders*, exhibiting disrupted functional connectivity patterns.  
   - This setting evaluates **cross-neurocognitive transfer** performance.

Each experiment follows the same **three-stage training procedure** described below.

---

### 1. Pre-training

A baseline model is pre-trained using data from one disorder within each pair (e.g., BD or ASD).

- **Code examples:**  
  - `BD_baseline.py` for BD  
  - `ASD_baseline.py` for ASD  
  - `AD_baseline.py` and `MDD_baseline.py` for complementary domains  
- **Model saved as:** `best-Model`

---

### 2. Source Prompt Training

Using the pre-trained baseline model on the source domain, we train:

- **Source mask prompt**  
- **Source disorder-specific prompts**

- **Code examples:**  
  - `source_prompt_BD.py`, `source_prompt_MDD.py`  
  - `source_prompt_ASD.py`, `source_prompt_AD.py`  
- **Models saved as:** `Model-1/` folders (per source disorder)

---

### 3. Target Prompt Training

Using the pre-trained baseline model and source prompts, we further train the **target prompt** on the complementary domain.

- The **target mask prompt** is initialized from the **source mask prompt**.  
- The **source disorder-specific prompts** and **cross-disorder attention** are used to generate **adaptive instance-level prompts**.

- **Code examples:**  
  - `target_prompt_MDD_from_BD.py`, `target_prompt_BD_from_MDD.py`  
  - `target_prompt_AD_from_ASD.py`, `target_prompt_ASD_from_AD.py`  
- **Models saved as:** `M-Model/`, `M-prompt1/` folders

---

## Comparison Experiments

**Fine-tuning:**  
For comparison, the pre-trained model is fine-tuned on the target domain by updating **all** model parameters, without employing any prompt learning mechanism.

- **Code examples:**  
  - `fine_MDD_from_BD.py`, `fine_BD_from_MDD.py`  
  - `fine_AD_from_ASD.py`, `fine_ASD_from_AD.py`

---

## Summary

This experimental framework enables a systematic evaluation of **cross-disorder transfer learning** between related mental and neurological disorders:

- **Affective-level transfer:** BD ↔ MDD  
- **Neurocognitive-level transfer:** ASD ↔ AD  

The experimental results, published in *IEEE JBHI (2025)*, demonstrate the **efficiency, scalability, and adaptability** of the proposed **Prompt Learning-based Transfer Learning framework** across heterogeneous brain disorder datasets.
