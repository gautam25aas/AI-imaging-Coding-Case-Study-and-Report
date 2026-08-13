# AI-imaging-Coding-Case-Study-and-Report
End-to-end microscopy image analysis pipeline combining multimodal LLM prompting, classical Otsu segmentation, PyTorch U-Net segmentation, region-based feature extraction, structured JSON generation, and robustness evaluation.
# Multimodal and Hybrid Image Analysis of Synthetic Fluorescence Microscopy Images

This repository contains the code, experiments, results, and report for an end-to-end microscopy image analysis pipeline developed for the **Data Analysis with AI** module at the **University of Hertfordshire**.

The project combines:

- Multimodal Vision-Language Models (VLMs)
- Prompt engineering
- Classical image segmentation using Otsu thresholding
- Morphological image processing
- Region-based feature extraction using `scikit-image`
- Deep learning segmentation using a PyTorch U-Net
- Structured LLM interpretation using JSON
- Hybrid segmentation-to-LLM analysis
- Robustness testing
- Loss-function comparison

---

## Project Overview

The aim of this project is to build and evaluate a hybrid microscopy image analysis pipeline that combines computer vision, deep learning, and local Large Language Models.

The supplied synthetic fluorescence microscopy dataset contains:

- 80 training images
- 20 validation images
- 12 unseen test images

All images were converted to grayscale and resized to:

`256 × 256`

The overall workflow is:

`Microscopy Image → Preprocessing → Segmentation → Feature Extraction → Structured LLM Interpretation → JSON / Narrative / CSV`

---

## Task 1 — Data Preparation and Direct VLM Description

The dataset was loaded, converted to grayscale, and resized to `256 × 256`.

Exploratory Data Analysis included:

- Representative microscopy images
- Global grayscale intensity distribution
- Selection of a representative training image

The representative image was:

`train_033.png`

Its mean intensity was:

`0.03599`

The overall training-set mean intensity was approximately:

`0.03595`

Therefore, `train_033.png` was selected because its mean intensity was closest to the dataset mean.

### Direct Vision-Language Model Analysis

A local multimodal model was queried using Ollama.

The assignment originally specified:

`llama3.2-vision`

However, the local Ollama installation produced the following model-loading error:

`unknown model architecture: 'mllama'`

Therefore, the local multimodal model used for Task 1 was:

`qwen2.5vl:3b`

The same prompting and evaluation procedure was retained.

### Naive Prompt

```text
Describe this microscopy image.
```

The naive model response incorrectly described the image as an electron microscopy image and speculated about viruses and microorganisms.

This demonstrated the hallucination risk of open-ended multimodal prompting.

### Optimised Prompt

The improved prompt constrained the model to:

- remain descriptive rather than diagnostic
- avoid inventing tissue type, organ, stain, species, or disease
- return `"uncertain"` when evidence was insufficient
- return valid JSON only
- separate visible observations from unsupported interpretation

The structured output included:

```json
{
  "modality": "fluorescence microscopy",
  "tissue_type": "uncertain",
  "notable_features": [
    "grayish oval shapes scattered across the field of view"
  ],
  "image_quality": {
    "overall": "good",
    "contrast": "high",
    "noise": "low",
    "issues": []
  }
}
```

This demonstrated that explicit role, constraints, uncertainty handling, and structured JSON significantly reduced unsupported claims.

---

## Task 2 — Classical Segmentation and Numbers-First LLM Interpretation

Classical segmentation was performed using:

- Otsu thresholding
- Morphological cleanup
- Small-object removal
- Hole filling
- Connected-component labelling
- `regionprops_table`

Extracted measurements included:

- Area
- Eccentricity
- Solidity
- Mean intensity
- Equivalent diameter
- Perimeter

For the representative image:

- Number of objects: `24`
- Mean area: `172.75 pixels`
- Mean eccentricity: `0.703`
- Mean solidity: `0.955`
- Foreground fraction: `0.063`
- Density class: `moderate`
- Shape regularity: `moderately_regular`

### Otsu Performance on Representative Image

- Dice: `0.9803`
- IoU: `0.9613`

### Mean Otsu Validation Performance

- Mean Dice: `0.9782 ± 0.0027`
- Mean IoU: `0.9574 ± 0.0052`

---

## Numbers-First LLM Interpretation

Unlike Task 1, the LLM did not receive the microscopy image.

It received only numerical measurements calculated by Python.

The prompt explicitly prevented the LLM from inventing:

- Tissue type
- Disease
- Organ
- Cell type
- Stain
- Species
- Acquisition method

The deterministic Python-derived values were not allowed to be modified.

Example structured response:

```json
{
  "description": "The image contains 24 objects with moderate density and moderately regular shapes, indicating a relatively uniform distribution of features.",
  "n_objects": 24,
  "density_class": "moderate",
  "shape_regularity": "moderately_regular",
  "quality_flag": "uncertain"
}
```

This numbers-first approach was more auditable because the important measurements were calculated deterministically rather than generated by the LLM.

---

## Task 3 — U-Net Segmentation

A compact U-Net was implemented using PyTorch.

### Model Configuration

- Framework: PyTorch
- Trainable parameters: `483,153`
- Training images: `80`
- Validation images: `20`
- Batch size: `4`
- Optimizer: Adam
- Learning rate: `0.001`
- Epochs: `15`
- Main loss: BCE + Dice Loss
- Device: CPU

### Final Training Results

At epoch 15:

- Training loss: `0.2267`
- Validation loss: `0.2210`
- Training Dice: `0.9957`
- Validation Dice: `0.9960`
- Validation IoU: `0.9921`

### U-Net vs Otsu

| Method | Mean Dice | Mean IoU |
|---|---:|---:|
| Otsu | 0.9782 | 0.9574 |
| U-Net | 0.9960 | 0.9921 |
| Improvement | +0.0178 | +0.0348 |

The U-Net achieved a higher Dice score than Otsu on all 20 validation images.

The largest observed improvement occurred on:

`val_019.png`

with approximately:

- Otsu Dice: `0.9757`
- U-Net Dice: `0.9979`

The results demonstrate that the U-Net learned spatial and boundary information that could not be represented by a single global intensity threshold.

---

## Validation Visualisation

Validation examples include:

`Input Image | Ground Truth Mask | U-Net Prediction`

Examples were selected from best, middle, and worst validation cases rather than displaying only the strongest predictions.

---

## Extra Credit — Loss Function Comparison

Three loss configurations were compared using matched short training runs:

- BCE
- Dice Loss
- BCE + Dice Loss

For this dataset and training setup, BCE achieved the highest short-run validation Dice:

`0.9954`

This result is specific to this experiment and does not imply that BCE is universally superior for segmentation problems.

---

## Task 4 — Hybrid Pipeline

The trained U-Net was applied to all 12 unseen test images.

The complete pipeline was:

```text
Test Image
    ↓
U-Net Prediction
    ↓
Binary Mask
    ↓
Morphological Cleanup
    ↓
Connected Components
    ↓
Region-Based Feature Extraction
    ↓
Deterministic Source Record
    ↓
Constrained Local LLM
    ↓
Validated JSON + Narrative
    ↓
Pandas DataFrame
    ↓
CSV Export
```

Python calculated the source-of-truth fields:

- `image_id`
- `n_objects`
- `mean_area`
- `density_class`
- `quality_flag`

The LLM was not allowed to change these fields.

Each LLM response was:

- parsed
- validated
- checked against deterministic values

If a response was invalid or altered source values, a deterministic fallback could be used.

---

## Task 4 Results

The hybrid pipeline processed:

`12 / 12 unseen test images`

Structured-output validation success:

`100%`

### Test-Set Summary

- Mean detected objects: `20.67`
- Mean object area: `254.83 pixels`
- Low-density images: `6`
- Moderate-density images: `6`
- Good project-defined quality flag: `12 / 12`

The density and quality categories used in this project are analytical rules created for the assignment and are not clinically validated categories.

### Example Structured Output

```json
{
  "image_id": "test_000.png",
  "n_objects": 8,
  "mean_area": 191.75,
  "density_class": "low",
  "quality_flag": "good",
  "narrative": "The image contains 8 objects with a mean area of 191.75 pixels..."
}
```

---

## Extra Credit — Robustness Testing

Robustness was evaluated by applying heavy Gaussian blur to one test image.

Blur configuration:

`Gaussian sigma = 4`

The corrupted image was passed through the complete pipeline.

### Clean vs Blurred Prediction

- Mask Dice: `0.7110`
- Mask IoU: `0.5516`

Feature changes included:

| Feature | Clean | Blurred |
|---|---:|---:|
| Number of objects | 8 | 7 |
| Mean area | 191.75 | 397.29 |
| Mean eccentricity | 0.6350 | 0.5344 |
| Mean solidity | 0.9668 | 0.9541 |
| Mean intensity | 0.2269 | 0.1119 |
| Foreground fraction | 0.0234 | 0.0424 |

The earliest clear failure was detected at the segmentation stage.

The altered segmentation then propagated into:

- Region measurements
- Object statistics
- Structured output
- Narrative generation

This demonstrates that downstream LLM interpretation can faithfully propagate incorrect upstream evidence even when the LLM itself is constrained.

---

## Direct VLM vs Numbers-First Interpretation

The direct VLM was more useful for describing visual appearance because it had access to the image.

However, the numbers-first approach was more trustworthy and auditable because the important measurements were calculated by Python.

The naive VLM demonstrated hallucination by incorrectly describing the image as electron microscopy and speculating about viruses and microorganisms.

The optimised prompt reduced this risk through:

- Explicit role definition
- Non-diagnostic constraints
- `"uncertain"` as an allowed response
- Strict JSON output
- Reduced freedom for unsupported interpretation

---

## Hallucination Risk and Safety

LLM hallucination may occur during:

- Direct image interpretation
- Open-ended biological descriptions
- Narrative generation
- Unsupported tissue or disease inference

Risk was reduced using:

- Structured JSON schemas
- Explicit non-diagnostic instructions
- Low-temperature generation
- Deterministic feature calculation in Python
- Locked source-of-truth values
- Output validation
- Fallback generation
- Explicit uncertainty

The structured JSON record was retained as the source of truth rather than relying on free-text narratives.

---

## Clinical Trustworthiness

This system should not be considered clinically validated.

The dataset is:

- Small
- Synthetic
- Limited in diversity

The high segmentation scores therefore demonstrate performance on this dataset only.

They do not demonstrate performance on real clinical microscopy data.

The most important improvement would be independent external validation using a large and diverse real microscopy dataset with expert annotations.

Such validation should include:

- Multiple laboratories
- Different microscopes
- Different imaging conditions
- Different biological samples
- Human expert review
- Uncertainty calibration
- Prospective evaluation

---

## Main Results

| Experiment | Result |
|---|---:|
| Otsu Mean Dice | 0.9782 |
| Otsu Mean IoU | 0.9574 |
| U-Net Mean Dice | 0.9960 |
| U-Net Mean IoU | 0.9921 |
| Dice Improvement | +0.0178 |
| IoU Improvement | +0.0348 |
| Best Short Loss Ablation Dice | 0.9954 |
| Task 4 JSON Validation | 100% |
| Robustness Dice | 0.7110 |
| Robustness IoU | 0.5516 |

---

## Technologies Used

- Python
- Jupyter Notebook
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-image
- PIL
- Ollama
- Qwen2.5-VL 3B
- Llama 3.2
- JSON
- CSV

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/gautam25aas/AI-imaging-Coding-Case-Study-and-Report.git
```

```bash
cd AI-imaging-Coding-Case-Study-and-Report
```

### 2. Install Python Packages

```bash
pip install numpy pandas matplotlib scikit-image pillow requests torch torchvision
```

### 3. Install Ollama Models

The project used:

```text
qwen2.5vl:3b
```

for multimodal image interpretation and:

```text
llama3.2:latest
```

for text-based structured interpretation.

Example commands:

```bash
ollama pull qwen2.5vl:3b
```

```bash
ollama pull llama3.2
```

### 4. Start Ollama

Make sure the local Ollama server is running before executing the LLM-related notebook cells.

### 5. Run the Notebook

Open the Jupyter notebook and execute the cells sequentially.

---

## Important Note About Llama 3.2 Vision

The assignment specified Llama 3.2 Vision.

It was installed and tested locally, but the available Ollama environment returned:

```text
unknown model architecture: 'mllama'
```

Therefore, `qwen2.5vl:3b` was used as the local multimodal substitute while retaining the same prompt-engineering and evaluation methodology.

---

## Output Files

The project produces:

- Exploratory microscopy visualisations
- Intensity histograms
- Otsu segmentation masks
- Region-feature tables
- Training and validation curves
- U-Net segmentation predictions
- Dice and IoU metrics
- Structured LLM JSON
- Narrative summaries
- Test-set CSV files
- Robustness visualisations
- Final assignment report

---

## Key Design Principle

A central design principle of this project is:

> Use deterministic computer vision and Python calculations for measurements, and use the LLM primarily for constrained interpretation and communication.

This approach improves:

- Auditability
- Reproducibility
- Structured validation
- Hallucination control
- Traceability of numerical results

---

## Limitations

The main limitations are:

- Small synthetic dataset
- Limited image diversity
- No external clinical validation
- Dependence on segmentation quality
- Local LLM outputs can still hallucinate
- Density and quality labels are project-defined
- High validation performance may not generalise to real microscopy

---

## Conclusion

This project demonstrates an end-to-end hybrid microscopy analysis pipeline combining classical image processing, deep learning segmentation, and local LLM interpretation.

The U-Net achieved strong segmentation performance with:

- Dice = `0.9960`
- IoU = `0.9921`

and outperformed Otsu segmentation across the full validation set.

The numbers-first LLM approach was more auditable than direct multimodal interpretation because numerical values were generated deterministically and validated before narrative generation.

The robustness experiment also demonstrated an important limitation: corruption introduced at the image level can affect segmentation, feature extraction, and ultimately the generated narrative.

Overall, the project shows the value of combining deterministic image analysis with constrained LLM interpretation while maintaining structured outputs as the source of truth.

---

## References

1. Otsu, N. (1979). *A Threshold Selection Method from Gray-Level Histograms*. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66.

2. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015, 234–241.

3. Korabel, N. (2026). *Lecture 3: Multimodal LLMs*. Data Analysis with AI, University of Hertfordshire.

4. Bhattacharya, S. (2026). *Prompt Engineering, Evaluation & Code Generation with Large Language Models*. Data Analysis with AI, University of Hertfordshire.

---

## GitHub Repository

https://github.com/gautam25aas/AI-imaging-Coding-Case-Study-and-Report

---

## Author

**Gautam**

University of Hertfordshire  
Data Analysis with AI
