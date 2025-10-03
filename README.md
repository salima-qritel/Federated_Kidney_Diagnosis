# 🩺 Federated Kidney Diagnosis 

A federated learning framework for the early detection of Chronic Kidney Disease (CKD), combining Flower, Docker, and a full monitoring stack with Prometheus and Grafana.

## 🌍 Overview

This project simulates a federated environment where multiple institutions (e.g., hospitals, laboratories) collaborate to train a predictive model for CKD **without sharing sensitive patient records.**.

Core components:

* 🧠 **Flower** → FL framework
* 🐍 **Scikit-learn** & **MLPClassifier** → local training
* 🐳 **Docker** & `docker-compose` → containerization and orchestration
* 📊 **Prometheus & Grafana** → monitoring pipeline and dashboards

---

## 📂 Repository Layout

```
federated-kidney-diagnosis/
├── client1/            # First FL client
|   ├──app/               
│      ├── client.py
|      ├── utils.py           
│   ├── dataset
|      ├── client1_data.csv
│   |── Dockerfile
│   └── requirements.txt
├── client2/            # Second FL client
|   ├──app/               
│      ├── client.py
|      ├── utils.py           
│   ├── dataset
|       ├── client2_data.csv
│   |── Dockerfile
│   └── requirements.txt
├── server/                # FL server logic
│   ├── server.py
|   |── Dockerfile
│   └── requirements.txt
├── monitoring/           # Monitoring configuration
│   ├── prometheus/
│   │   └── prometheus.yml
│    
├── docker-compose.yml     # Multi-container orchestration
└── README.md             
```

---

## ⚙️ Workflow

**Clients:** Each client trains locally on its private CKD dataset.

**Server:** Runs Flower with FedAvg to aggregate model updates.

**Monitoring:** Metrics are pushed to Prometheus and visualized in Grafana dashboards.

---

## 🛠️ Setup & Execution

### 1. Clone the repository

```bash
git clone https://github.com/OumaimaTF/federated-kidney-diagnosis.git
cd federated-kidney-diagnosis
```

### 2. Launch system

```bash
docker-compose up --build
```

> This starts the server, clients, and monitoring stack inside isolated containers.

### 3. Dashboardss

* 📈 Prometheus: [http://localhost:9090](http://localhost:9090)
* 📋 Grafana: [http://localhost:3000](http://localhost:3000)

  * Login: `admin` / `admin`

---

## 📊 Grafana Dashboard Example

> Visualize training loss, accuracy, number of rounds, etc.

![Grafana Screenshot](./monitoring/grafana/png/grafana.png)

---

## 📚 Dataset

Dataset used: [Chronic Kidney Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/rabieelkharoua/chronic-kidney-disease-dataset-analysis/data)

---


## 🔄 Execution Pipeline

This is a brief overview of how the system runs:

1. **Infrastructure Startup** via Docker Compose:
   - Starts the Flower server
   - Launches two FL clients simulating medical centers
   - Starts Prometheus, Grafana, and Pushgateway for monitoring

2. **Federated Learning Setup**:
   - Each client registers with the server
   - The server orchestrates training rounds using the FedAvg aggregation algorithm

3. **Local Training & Aggregation**:
   - Each client trains an `MLPClassifier` on its local CKD dataset
   - Model weights are sent to the server for aggregation

4. **Metrics Monitoring**:
   - Clients and server push custom training metrics via Prometheus Pushgateway
   - Grafana visualizes metrics like accuracy, loss, and round number in real time

5. **Shutting Down**:
   - To gracefully stop all containers:
     ```bash
     docker-compose down
     ```

---


## 🏗️ Tech Stack

* [Flower](https://flower.dev/)
* [Scikit-learn](https://scikit-learn.org/)
* [Docker](https://www.docker.com/)
* [Prometheus](https://prometheus.io/)
* [Grafana](https://grafana.com/)

---

## 🧑‍💻 Author

**Salima Qritel**

* GitHub: [@salima-qritel](https://github.com/salima-qritel)
  

---
