### ***\*Title\*******\*:\*******\*An Efficient Transfer Learning with Prompt Learning for Brain Disorders Diagnosis\****



 

### ***\*Data Preprocessing\****

The time-series data is divided using a sliding window mechanism, followed by the calculation of the Pearson Correlation Coefficient (PCC). The PCC matrix for each segment is then processed using an upper-triangular transformation, resulting in 6670 elements.
Corresponding code: data_process.py, data_process_dimension_transducer.py

### ***\*Experimental Setup\****

In this experiment, BD (Bipolar Disorder) is considered as the source domain, and MDD (Major Depressive Disorder) is treated as the target domain.

#### ***\*1. Pre-training\****

The baseline model is pre-trained using data from the source domain (BD).
Corresponding code: BD_baseline.py
Model saved as: best-Model

#### ***\*2. Source Prompt Training\****

Using the pre-trained baseline model on source domain data, we train the following:

· Source mask prompt

· Source disorder-specific prompts

Corresponding code: source_prompt_BD.py
Models saved as: Model-1 folders

#### ***\*3. Target Prompt Training\****

Using the pre-trained baseline model and source prompts, we train the target prompt.

· The source mask prompt is used to initialize the target mask prompt.

· The source disorder-specific prompts and cross-disorder attention are utilized to generate the adaptive instance-level prompts.

Corresponding code: target_prompt_MDD.py
Models saved as: M-Model, M-prompt1 folders



### ***\*Comparison Experiments:\****

**Fine-tuning**: The pre-trained model is fine-tuned on the target domain by updating all model parameters.
Code: fine_MDD.py

 
