
# Plant Leaf Disease Classification & Smart Advisory System

A deep learning–based web application that detects plant leaf diseases and provides treatment and prevention recommendations using confidence-aware predictions and seasonal context.

---

## Project Overview

This project implements a DenseNet121-based Convolutional Neural Network (CNN) trained on 20,000+ images from the PlantVillage dataset for plant disease classification.

In addition to classification, the system includes a Smart Advisory Engine that:

* Provides treatment recommendations
* Suggests prevention strategies
* Incorporates seasonal context
* Uses prediction confidence to improve advisory reliability

The system is deployed as a Flask web application for real-time disease diagnosis via image upload.

---

## Model Details

* Architecture: DenseNet121 (Transfer Learning)
* Framework: PyTorch
* Dataset: PlantVillage (~20K+ images)
* Accuracy Achieved: ~92%
* Image Processing: OpenCV
* Deployment: Flask

---

## Features

* Real-time leaf image upload
* Disease prediction with confidence score
* Confidence-aware advisory system
* Treatment and prevention recommendations
* Seasonal context integration
* Flask-based web deployment

---

## Tech Stack

* Python
* PyTorch
* OpenCV
* DenseNet121
* Flask
* HTML/CSS

---

## Project Structure

```
├── app.py
├── model/
│   ├── train.py
│   ├── densenet_model.pth
├── static/
│   ├── uploads/
├── templates/
│   ├── index.html
├── utils/
│   ├── preprocessing.py
│   ├── advisory_engine.py
└── README.md
```

---

## How It Works

1. User uploads a plant leaf image.
2. The image is preprocessed using OpenCV.
3. The DenseNet121 model predicts the disease class.
4. A confidence score is calculated.
5. The advisory engine:

   * Checks confidence threshold
   * Integrates seasonal context
   * Generates treatment and prevention advice
6. Results are displayed on the web interface.

<img width="959" height="450" alt="plant" src="https://github.com/user-attachments/assets/9ad0ef90-aa28-40a3-9394-fe24f5533bb5" />
<img width="265" height="411" alt="plant2" src="https://github.com/user-attachments/assets/6c59fbba-e9d0-4032-9817-d2ee56e5bbc0" />


---

## Installation and Setup

### Clone the Repository

```
git clone https://github.com/your-username/plant-leaf-disease-classification.git
cd plant-leaf-disease-classification
```

### Create Virtual Environment

```
python -m venv venv
source venv/bin/activate
```

On Windows:

```
venv\Scripts\activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Run the Application

```
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## Model Performance

* Accuracy: ~92%
* Dataset Size: 20K+ images
* Architecture: DenseNet121

---

## Future Improvements

* Cloud deployment (AWS / GCP / Render)
* Grad-CAM visualization
* Multi-language advisory support
* Weather API integration
* Mobile application integration

---

