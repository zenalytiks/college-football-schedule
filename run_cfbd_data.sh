#!/bin/bash

# Set paths - UPDATE THESE WITH YOUR ACTUAL PATHS
APP_DIR="/root/college-football-schedule"
VENV_DIR="/root/college-football-schedule/venv"
LOG_FILE="$APP_DIR/logs/cfbd_data_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory if it doesn't exist
mkdir -p "$APP_DIR/logs"

# Create data directory if it doesn't exist
mkdir -p "$APP_DIR/data"

# Change to script directory
cd "$APP_DIR"

# Log start time
echo "$(date): Starting CFBD data collection..." >> "$LOG_FILE"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Check if virtual environment was activated successfully
if [ $? -ne 0 ]; then
    echo "$(date): Failed to activate virtual environment at $VENV_DIR" >> "$LOG_FILE"
    exit 1
fi

# Run the Python script and capture output
python cfbd_data_reader.py >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

# Deactivate virtual environment
deactivate

# Log completion status
if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date): CFBD data collection completed successfully" >> "$LOG_FILE"
    
    # Optional: Log file size for monitoring
    if [ -f "$APP_DIR/data/cfbd_data.csv" ]; then
        FILE_SIZE=$(wc -l < "$APP_DIR/data/cfbd_data.csv")
        echo "$(date): CSV file contains $FILE_SIZE lines" >> "$LOG_FILE"
    fi
else
    echo "$(date): CFBD data collection failed with exit code $EXIT_CODE" >> "$LOG_FILE"
fi

# Optional: Keep only last 30 log files to prevent disk space issues
find "$APP_DIR/logs" -name "cfbd_data_*.log" -type f -mtime +30 -delete

exit $EXIT_CODE
