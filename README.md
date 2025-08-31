# 🧠 FPT+ — Fine-Grained Prompt Tuning for Medical Image Classification

This project implements **FPT+**, a memory-efficient method for **high-resolution medical image classification**, leveraging **parameter-efficient transfer learning (PETL)**. The model integrates a **pre-trained Vision Transformer (ViT)** with a lightweight side network using fine-grained prompts, aiming to significantly reduce trainable parameters without compromising accuracy.

---

## 🚀 Key Features

- ⚙️ **Pre-trained Vision Transformer** backbone for feature extraction
- 📦 **Fine-grained prompt tuning** to inject task-specific signals without full model finetuning
- 🧠 **Side network** for capturing high-resolution local features
- 💾 Optimized for **parameter efficiency**, ideal for low-resource environments
- 🧪 Applicable to various medical image datasets (e.g., X-ray, CT, histopathology)

---

## 🛠️ Tech Stack

| Tool/Library       | Role                                      |
|--------------------|-------------------------------------------|
| **Python 3.10+**    | Programming language                      |
| **PyTorch**        | Deep learning framework                   |
| **HuggingFace Transformers** | For loading Vision Transformer  |
| **OpenCV / PIL**   | Image preprocessing                       |
| **Matplotlib/Seaborn** | Visualizations                        |

---

## 🧬 Model Overview

- **Backbone**: Pre-trained ViT (e.g., `vit-base-patch16-224`)
- **Prompt Tuning**: Injects trainable tokens at intermediate transformer layers
- **Fusion Module**: Combines features from ViT and side network
- **Classifier Head**: Final MLP for image-level classification

## Link to Dataset 

- **https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia**
