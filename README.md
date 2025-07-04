# 🧠 VisionFlow: Modular CV Inference API

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-009688.svg)
![Docker Support](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

An industrial-grade computer vision microservice designed for model inference. Built with **FastAPI**, optimized for **Docker environments**, and structured to scale across multiple vision tasks like classification and object detection.

---

## 🗂 Overview

This system is optimized for:

- Serving multiple CV models via HTTP APIs
- Handling large-scale requests with configurable **rate limiting**
- Plug-and-play extensibility for new tasks, models, or strategies

Ideal for deploying computer vision inference systems in production environments.

---

## 📌 Index

- [Key Capabilities](#key-capabilities)
- [System Design](#system-design)
- [How to Launch](#how-to-launch)
- [File Guide](#file-guide)
- [Environment Settings](#environment-settings)
- [API Interface](#api-interface)
- [Throttle Controls](#throttle-controls)
- [Plug-in Extensions](#plug-in-extensions)
- [Validation & Testing](#validation--testing)
- [Production Deployment (Docker)](#production-deployment-docker)
- [Development & Contribution](#development--contribution)
- [License](#license)

---

## ✅ Key Capabilities

- 🧠 **Multi-task Inference**: Use models like YOLO or EfficientNet for classification, detection, and custom use-cases.
- 🧩 **Modular Expansion**: Swap models, modify pipelines, or insert new preprocess/postprocess steps.
- 🕵️ **Custom Strategies**: Tailor prediction logic using modular "strategy" classes.
- 🚥 **Rate Management**: Redis-based request limiting prevents abuse and stabilizes access.
- 📦 **Container-First**: Docker images for simplified setup and deployment.
- 🧪 **Full Test Coverage**: Includes unit tests, integration tests, and API contract checks.

---

## 🧱 System Design

```
project-root/
│
├── src/
│   ├── conf/                 ← YAML/ENV configurations
│   ├── schema/               ← API data models
│   ├── service/
│   │   ├── constants/        ← Image-related constants
│   │   ├── models/           ← Model wrappers (YOLO, EfficientNet)
│   │   ├── pipelines/        ← Execution logic (pre → model → post)
│   │   ├── processing/       ← Preprocessing and postprocessing modules
│   │   ├── prediction_strategies/
│   │   ├── tools.py
│   │   └── service.py        ← FastAPI app entry point
│   ├── settings/
│   └── logger.py
```

---

## ⚡ How to Launch

### Option 1: Use Docker (Production-Ready)

```bash
git clone https://github.com/your-org/vision-inference-pipeline.git
cd vision-inference-pipeline
cp .env.example .env   # update settings
docker-compose up --build
```

📍 App will be live at `http://localhost:8000`  
📚 Swagger UI: `http://localhost:8000/docs`

---

### Option 2: Local Dev Setup (with Poetry)

```bash
git clone https://github.com/your-org/vision-inference-pipeline.git
cd vision-inference-pipeline
poetry install
cp .env.example .env
poetry shell
python src/run_service.py
```

---

## 📂 File Guide

| Folder | Purpose |
|--------|---------|
| `conf/` | Config settings for the models and pipelines |
| `models/` | Detection & classification models (YOLO, EfficientNet) |
| `processing/` | Input/output transformation layers |
| `pipelines/` | Flow logic for each CV task |
| `prediction_strategies/` | Rotation-based or fallback logic |
| `settings/` | App-level constants and .env loader |

---

## ⚙️ Environment Settings

All runtime behavior (model path, allowed hosts, Redis setup, etc.) is configured via the `.env` file and loaded by the service settings module.

---

## 🔌 API Interface

Interactive API documentation is available at:

```
http://localhost:8000/docs
```

Typical endpoints:
- `POST /predict/classify`
- `POST /predict/detect`
- `GET /health`

---

## 🛡️ Throttle Controls

Enable and configure Redis-based rate limiting to protect API routes. Useful in high-traffic or public deployments.

---

## 🧰 Plug-in Extensions

You can easily expand the system by:

- 📌 Adding new models (drop them under `models/`)
- ✂️ Creating new preprocessors
- 🧠 Building post-processing logic
- 🔁 Crafting new prediction strategies

All extensions can be auto-registered using decorators or the factory pattern.

---

## 🧪 Validation & Testing

Run tests using:

```bash
pytest
```

Covers unit tests, route checks, and pipeline validation.

---

## 📦 Production Deployment (Docker)

Use the provided Dockerfile and `docker-compose.yml` for reproducible, containerized deployment. Compatible with major container orchestration tools.

---

