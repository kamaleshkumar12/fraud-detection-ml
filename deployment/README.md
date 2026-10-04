# Fraud Detection API

Run locally (Docker Desktop venum):

    docker build -t fraud-api .
    docker run -p 8000:8000 fraud-api

Test:      python test_api.py http://localhost:8000
API docs:  http://localhost:8000/docs
Drift:     python monitor.py sample_recent_transactions.csv
