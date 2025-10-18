# An Efficient Transfer Learning with Prompt Learning for Brain Disorders Diagnosis

## Data Preprocessing

The time-series data is divided using a sliding-window mechanism, followed by the calculation of the Pearson Correlation Coefficient (PCC).  
For each segment, the PCC matrix is vectorized by taking the upper-triangular part, yielding **6,670** elements.

In addition, for the **AD (Alzheimer’s Disease)** dataset, we performed **data augmentation** to alleviate the limited-sample problem and enhance model generalization.

- Corresponding code:  
  - `data_process.py`  
  - `data_process_dimension_transducer.py`  
  - `AD_data_process.py` *(for data augmentation in AD)*

---

## Experimental Setup

We design two cross-disorder transfer learning settings to evaluate the generalization and adaptability of the proposed framework:

1. **BD ↔ MDD**: Bipolar Disorder and Major Depressive Disorder  
   - Both belong to *affective disorders*, sharing overlapping symptoms but distinct neural patterns.  
   - This setting evaluates **cross-affective transfer** performance.

2. **ASD ↔ AD**: Autism Spectrum Disorder and Alzheimer’s Disease  
   - Both belong to *neurodevelopmental and neurodegenerative disorders*, exhibiting disrupted functional connectivity patterns.  
   - This setting evaluates **cross-neurocognitive transfer** performance.

Each experiment follows the same three-stage procedure described below.

---

### 1. Pre-training

A baseline model is pre-trained using data from one disorder within each pair (e.g., BD or ASD).

- Corresponding code examples:  
  - `BD_baseline.py` for BD  
  - `ASD_baseline.py` for ASD  
  - `AD_baseline.py` and `MDD_baseline.py` for complementary domains  
- Model saved as: `best-Model`

---

### 2. Source Prompt Training

Using the pre-trained baseline model on the source domain, we train:

- **Source mask prompt**  
- **Source disorder-specific prompts**

- Corresponding code examples:  
  - `source_prompt_BD.py`, `source_prompt_MDD.py`  
  - `source_prompt_ASD.py`, `source_prompt_AD.py`  
- Models saved as: `Model-1/` folders (per source disorder)

---

### 3. Target Prompt Training

Using the pre-trained baseline model and source prompts, we train the **target prompt** on the complementary domain.

- The **target mask prompt** is initialized from the **source mask prompt**.  
- The **source disorder-specific prompts** and **cross-disorder attention** are used to generate **adaptive instance-level prompts**.

- Corresponding code examples:  
  - `target_prompt_MDD_from_BD.py`, `target_prompt_BD_from_MDD.py`  
  - `target_prompt_AD_from_ASD.py`, `target_prompt_ASD_from_AD.py`  
- Models saved as: `M-Model/`, `M-prompt1/` folders

---

## Comparison Experiments

**Fine-tuning:**  
The pre-trained model is fine-tuned on the target domain by updating **all** model parameters, without using prompts.

- Corresponding code examples:  
  - `fine_MDD_from_BD.py`, `fine_BD_from_MDD.py`  
  - `fine_AD_from_ASD.py`, `fine_ASD_from_AD.py`

---

## Summary

This setup enables comprehensive evaluation of **cross-disorder transfer learning** between related mental and neurological disorders:

- **Affective-level transfer:** BD ↔ MDD  
- **Neurocognitive-level transfer:** ASD ↔ AD  

It verifies the efficiency and adaptability of the proposed **Prompt Learning framework** across heterogeneous brain disorder datasets.
