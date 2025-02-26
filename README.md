# 🚨🤖 ddos-detector
Semester project for Advanced Data Analysis Methods Laboratory

Team members: 

- Laszlo Patrik Abrok (JPWF8N)
- Janos Hudak (CP4EIQ)

## ℹ️ Project description

This is our semester project for the [VITMMB10 Advanced Data Analysis Methods Laboratory](https://portal.vik.bme.hu/kepzes/targyak/VITMMB10/en/) course. It leverages real-life network data collected by [SmartComLab](https://smartcomlab.tmit.bme.hu/) to classify DDoS attacks with high precision.

## 📊 Data 

The network data is not included in this repository as it is currently under revision for publication.

## 🚀 Quick start

### Development 

#### Requirements 

- [Python v3.9](https://www.python.org/downloads/release/python-3921/) installed and on path (at least)  
- [Poetry v1.8](https://python-poetry.org/docs/#installation) installed and on path (at least)

[Activate the environment](https://python-poetry.org/docs/managing-environments/#activating-the-environment), e.g. in PowerShell:

```PowerShell
poetry shell
```

Install the dependencies:

```PowerShell
poetry install
```

### Deployment

#### Requirements 

- [Docker](https://www.docker.com/)

#### Option 1: Use the latest image from [Docker Hub](https://hub.docker.com/repository/docker/jahudak/ddos-detector/general) 

```PowerShell
docker run jahudak/ddos-detector:<tag>
```

#### Option 2: Build and deploy locally 

```PowerShell
docker build -t ddos-detector .
docker run ddos-detector
```

## 📜 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details or visit the [official Apache License 2.0 page](http://www.apache.org/licenses/LICENSE-2.0).