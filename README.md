# 🚢 Ship Detection using SAR Images – Binary Classifier

This project implements a **CNN-based ship detection model** using **SAR (Synthetic Aperture Radar)** satellite imagery.  
It performs **binary classification** to identify whether a ship is present in a given SAR image patch and outputs a **confidence score** indicating detection certainty.

The model is designed to run efficiently on **resource-constrained environments** such as onboard CubeSat systems, enabling real-time maritime monitoring.

---

## 🎯 Objective

- Input: 80×80 pixel SAR image patch  
- Output:  
  - **Binary label** → `Ship` / `No Ship`  
  - **Confidence score** → e.g., `Ship: 92.4%`

---

## 🛰️ Dataset

- Source: SAR satellite imagery  
- Format: Grayscale 80×80 images  
- Labels: Binary (1 = Ship, 0 = No Ship)

---

## 📁 Project Structure
