#### This project is for me to be able to learn and understand deep learning methods by trying to apply my skills for a real life project.

1. # Obtaining and processing the dataset into a suitable format

* Downloaded the full MFIB, DIBS and FuzDB datasets in XML format (on 18/05/2024). The files can be converted to the fasta format by running the `parse_mfib.py`, `parse_dibs.py` and `parse_fuzdb.py` respectively. 
    * parse_dibs.py: Parses an XML file from the DIBS database to extract protein sequences. This script reads a DIBS XML file, finds all protein chains marked as "Disordered", and writes their sequences to an output file in a FASTA-like format. The header for each sequence is composed of the protein name, accession number, and chain ID.
    * parse_mfib.py: Parses an XML file from the MFIB database to extract protein sequences. This script reads an MFIB XML file, finds all protein chains, and writes their sequences to an output file in a FASTA-like format. The header for each sequence is composed of the protein name, accession number, and chain ID.
    * parse_fuzdb.py: Parses an XML file from the FuzDB database to extract and process protein sequences. This script reads a FuzDB XML file, identifies all fuzzy regions within each entry, and merges any overlapping or adjacent regions. It then extracts the combined sequences and writes them to an output file in a FASTA-like format. The header for each sequence includes the protein name, entry ID, and the start and end positions of the combined region.
* Concatenate the dibs.fasta and mfib.fasta and cluster the comined sequences with a 70% identity threshold to reduce redundancy. Also, cluster the sequences from fuzdb.fasta. The sequences in the dibs.fasta and mfib.fasta constitute the disorder-to-order type while those from fuzdb.fasta constitutes the disorder-to-disorder type. The mfib+dibs dataset file is named as do_transition.fasta and the fuzdb dataset file is named 'dd_transition.fasta'. The final files after clustering, is named 'dd_transition_cdhit.fasta' and 'do_transition_cdhit.fasta'. The `dataset_clustering.ipynb` notebook has steps for concatenating and clustering the datasets, that can be run in a linux based environment.

* Downloaded the 'derived-binding_mode_disorder_to_order-mobi' and 'derived-binding_mode_disorder_to_disorder-mobi' in fasta format (16/7/2025) and using the `mobidb_dataset_processing.ipynb` notebook, the individual regions of interest in each sequence is obtained as seperate sequence in the fasta file. This involves:
    * Extracting the sequence with the coreesponding residue scores for only the condition of interest like 'derived-binding_mode_disorder_to_disorder-mobi' from the downloaded mobidb fasta file. The residue scores are 0 and 1, indicating whether the residue is true for the condition or not. The output is saved to a file.
    * From the previous file, based on the residue scores of 1, the residues forming a strech was extracted as a seperate sequence and the header indicated the sequence name of which the strech is part of along with type of disorder transition and the number indicating the region order. The output is saved to a file.
* The final processed fasta files named 'mobidb_do.fasta' and 'mobidb_dd.fasta was clustered using cd-hit with a 70% identity threshold to reduce redundancy and saved as 'mobidb_do_cdhit.fasta' and 'mobidb_dd_cdhit.fasta' respectively.
  

2. # Protein Language Model Transfer Learning for Sequence Classification (Type of transition of IDRs upon binding)

This project demonstrates the use of a pre-trained protein language model (PLM) for a binary classification task on whether IDRs (Intrinsically disordered regions) remain disordered or undergo a disorder-to-order transition upon binding. The notebook fine-tunes the ESM-2 model to classify protein sequences from different datasets.The name of the notebook is `PLM_Transfer_Learning_Sequence.ipynb`.

## Workflow

1.  **Data Preparation:**
    *   Loads the input sequences from the 'clustered_fasta_files' folder.
    *   Labels the clustered sequences: 0 for the 'do_transition_cdhit.fasta' (DIBS/MFIB) set and 1 for the 'dd_transition_cdhit.fasta' (FuzDB) set. Alternatively, using the 'mobidb_do_cdhit.fasta' and mobidb_dd_cdhit.fasta' respectively.
    *   Filters out sequences longer than 1000 amino acids.
    *   Removes any duplicate sequences.
    *   Splits the data into training and testing sets.

2.  **Model and Tokenization:**
    *   Uses the `facebook/esm2_t12_35M_UR50D` model, a protein language model from Hugging Face.
    *   Tokenizes the protein sequences to prepare them for the model.

3.  **Training:**
    *   To address class imbalance in the dataset, a custom `WeightedLossTrainer` is implemented with a `CrossEntropyLoss` function that uses pre-calculated class weights.
    *   The model is trained for 5 epochs with a learning rate of 2e-5 and a batch size of 4.
    *   The best model is selected based on the F1 score on the evaluation set.

4.  **Evaluation:**
    *   The model's performance is evaluated using F1, precision, and recall metrics.

5.  **Results:**
    *   The fine-tuned model and tokenizer are saved to the `PLM_Transfer_Sequence_Outputs/my_model_dir/` and `PLM_Transfer_Sequence_Outputs/tokenizer_dir/` directory.
    *   A log of the training process is saved to `PLM_Transfer_Sequence_Outputs/training_logs.csv`.

## Dependencies

To run this project, you need to install the following libraries:

*   `pandas`
*   `numpy`
*   `torch`
*   `scikit-learn`
*   `transformers`
*   `datasets`
*   `evaluate`
*   `huggingface_hub`
*   `fsspec`
*   `cd-hit`


The notebook will preprocess the data, train the model, and save the results.

3. # Protein Language Model Transfer Learning for Sequence Classification (Type of transition of IDRs upon binding) utilising LoRA (Low-Rank Adaptation)

Low-Rank Adaptation (LoRA) is a PEFT (arameter-efficient fine-tuning) method that decomposes a large matrix into two smaller low-rank matrices in the attention layers. This drastically reduces the number of parameters that need to be fine-tuned, minimizing the need for extensive resources. The name of the notebook is `PLM_Transfer_Learning_Sequence_LORA`.

* There is an addition of LoraConfig (from peft library) wich involves setting parameters that control how low-rank matrices are used to update model weights during fine-tuning, allowing for efficient adaptation of large language models. Key parameters include the rank (r), lora_alpha, lora_dropout, target_modules, and bias settings.
* There are some changes to the the model TrainingArgument like use of optim = "paged_adamw_8bit" to reduce memory usage. With 8-bit optimizers, large models can be finetuned with 75% less GPU memory without losing any accuracy compared to training with standard 32-bit optimizers. 
* The fine-tuned model and tokenizer are saved to the `PLM_Sequence_LORA_Outputs/my_model_dir/` and `PLM_Sequence_LORA_Outputs/tokenizer_dir/` directory.
* A log of the training process is saved to `PLM_Sequence_LORA_Outputs/training_logs.csv`.

## Additional Dependencies

To run this project, you need to install the following libraries:

*   `peft`
*   `bitsandbytes`

4. # Validation

The `Validation_datsets.ipynb` can be used to check the f1, precission and recall scores along with the confusion matrix for a trained model with datasets that can be used for validation. The previously saved models can be loaded from the saved directories like `PLM_Sequence_LORA_Outputs/my_model_dir/` and the tokens can be loaded from saved directories like `PLM_Sequence_LORA_Outputs/tokenizer_dir/`.