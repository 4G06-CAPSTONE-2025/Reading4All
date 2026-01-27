import os
import inspect
from datetime import datetime

# Exported Constants
LOG_FILE_PATH = os.getenv("LOGGER_FILE_PATH", "system_logs.txt")
LOG_TYPE = ["Event", "Error"]

class LogWriteException(Exception):
    pass

class Logger:
    @staticmethod
    def logEvent(eventMssg: str, logType: str):
        if logType not in LOG_TYPE:
            logType = "Unknown"

        # Find the caller of the logEvent function to add to the log event message
        caller_frame = inspect.stack()[1]
        caller_file = os.path.basename(caller_frame.filename)
        caller_line = caller_frame.lineno
        
        # Build log entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {logType.upper()} [{caller_file}:{caller_line}] - {eventMssg}\n"

        # Write entry to the log file
        try:
            with open(LOG_FILE_PATH, "a") as log_file:
                log_file.write(log_entry)
        except (IOError, OSError) as e:
            raise LogWriteException(f"Failed to write to {LOG_FILE_PATH}: {e}")