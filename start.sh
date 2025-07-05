#!/bin/sh
echo "Waiting for database..."
python init_db.py

if [ $? -eq 0 ]; then
    echo "Starting application..."
    uvicorn main:app --host 0.0.0.0 --port 8000 
else
    echo "Failed to initialize database"
    exit 1
fi