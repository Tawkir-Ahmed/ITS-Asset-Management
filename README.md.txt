# SmartWay ITS — Asset Intelligence Platform (Prototype)

A Streamlit-based decision-support dashboard for ITS asset inventory, operations KPIs, outage-risk scoring, renewal scenario planning, cross-system reconciliation, and report export.

> Note: This repository currently uses **synthetic demo data** to illustrate workflow and UI.

## Features
- **Summary**: KPI cards, asset mix, service-status chart (window-based), map overview, savings table
- **Inventory**: searchable asset table + map focus
- **Operations**: KPI rollups by asset type + maintenance/outage log
- **Scenario Planning**: 5–10 year replacement planning with budget constraints
- **Asset Map**: status markers (window-based) + quick counts
- **Data Quality**: cross-system reconciliation (type/location conflicts)
- **AI Risk Analysis**: per-asset predicted outage risk + top drivers (logistic regression demo)
- **Reports**: export PDF/DOCX/ZIP package

## Quick start
### 1) Create environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
