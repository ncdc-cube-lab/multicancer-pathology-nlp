# Development and Open-Source Release of a Multi-Cancer Module for Structuring Pathology Reports
A BERT-based NLP pipeline for structured information extraction from multi-cancer pathology reports with multi-institutional validation

## Overviw
- Framework: SQuAD-style Question Answering
- Models: ClinicalBERT
- Cancer types: Breast, Kidney, Thyroid, Liver, Colorectal

## Requirements
- Python 3.8+
- numpy==1.24.1
- pandas==1.4.2
- scikit-learn==1.0.2
- scipy==1.8.0
- tensorflow==2.11.0
- tensorflow-gpu==2.6.0
- keras==2.11.0
- torch==1.12.0
- transformers==4.18.0
- tokenizers==0.12.1
- huggingface-hub==0.5.1

## Usage
Follow the steps below to run the full pipeline:
1. Environment Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```
2. Model Download
Download the pretrained models required for inference
3. Data Preprocessing
Clean and preprocess the raw annotation data
(e.g., remove unnecessary records and adjust span positions):
```bash
python preprocessing/preprocess_annotations.py
```
4. Information Extraction
Convert the preprocessed data into SQuAD format and run the extraction model:
```bash
python pipeline/run_extraction.py
```
5.Evaluation
Evaluate the model predictions:
```bash
python evaluation/evaluate_predictions.py
```

## Model Checkpoints
Pre-trained ClinicalBERT checkpoints are available:
| Cancer Type | Download |
|-------------|----------|
| Breast      | [Google Drive](https://drive.google.com/drive/folders/1tS8eBFTEt9kba5oDmARfpZdatrXgGMsa?usp=sharing) |
| Kidney      | [Google Drive]() |
| Thyroid     | [Google Drive]() |
| Liver       | [Google Drive](https://drive.google.com/drive/folders/1fTQb0nRW9ol_IxTo29zft7SF4rjgdwj4?usp=sharing) |
| Colorectal  | [Google Drive](https://drive.google.com/drive/folders/1uuvfythOnqGKVWpkmCMSudFbPDqVs1nb?usp=sharing) |

## Ethics Statement
This study was approved by the Institutional Review Board of the National Cancer Center Korea (IRB No. NCC2025-0250).

## Contact




