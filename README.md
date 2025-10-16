# MorphoBot 🎭

*An intelligent Telegram bot for real-time demonstration of unpaired image-to-image translation models*

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)  
[![Telegram Bot API](https://img.shields.io/badge/Telegram%20Bot%20API-Latest-blue.svg)](https://core.telegram.org/bots/api)  
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)  

---

## 🚀 Features

- **Unpaired Image-to-Image Translation**  
  Live demonstrations of various existing approaches (e.g., CycleGAN, MUNIT, Schrodinger Bridges) applied to diverse datasets.
- **Gender-Swap Demo (Semi-Ready)**  
  Initial implementation using an unpaired gender-swap model; additional pretrained models (horse↔zebra, summer↔winter, etc.) in progress.
- **Data Preparation**  
  Extensive use of OpenCV and dlib for face detection, landmark extraction, and dataset preprocessing.
- **Bot Interface**  
  Telegram bot is used as a frontend to upload images and receive translated outputs.
- **Artifact Management**  
  Models and data managed via MLflow registry with project-specific tag set and lookup system designed around it.
- **PyTorch Ecosystem**  
  Core model implementations and inference workflows are built with pure Python and PyTorch ecosystem.
- **Containerized Deployment**  
  Entire stack shipped and orchestrated with Docker.

---

## 🎯 Demo
Try it now: [@UnpairedTranslationBot](https://t.me/[UnpairedTranslationBot])

---

## 🛠️ Technology Stack

- **Language & Frameworks**:  
  Python 3.10+, PyTorch, python-telegram-bot
- **Model Management**:  
  Models and artifacts are shipped with MLflow registry
- **Image Processing**:  
  Pillow, OpenCV, dlib for face detection, alignment, and preprocessing
- **Containerization**:  
  Docker, docker-compose for deployment
- **CI/CD & Testing**:  
  pytest, pytest-asyncio, pytest-mock; GitHub Actions for automated tests (coming in future)


## 📖 Usage

### Basic Translation Flow

1. `/start` — Begin interaction  
2. Upload an image  
3. Bot detects faces and preprocesses data (OpenCV + dlib)  
4. Bot runs the chosen PyTorch model via MLflow artifact  
5. Receive the translated image back in chat  

---

## 🚦 Roadmap

- **v0.1.0 (MVP)**  
  - Gender-swap unpaired translation demo  
  - Telegram bot interface  
  - MLflow integration  
- **v0.2.x (In Progress)**  
  - More models and datasets  
  - Production-ready deployment
  - CI/CD workflows, performance monitoring
- **v1.0.0 (Future)**  
  - Task selection and live demos
  - Benchmarking and comparison boards
---

## 📜 License

Apache License 2.0 © [@stankevich-mipt](https://github.com/stankevich-mipt)

---

## 👥 Contact

Questions? Reach out via GitHub Issues or Telegram: [@stankevich.as](https://t.me/stankevich.as)

---

*Built with PyTorch, MLflow, OpenCV, dlib, and Python-Telegram-Bot.*