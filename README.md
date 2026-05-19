services:
  - type: web
    name: analisis-em
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    plan: free
