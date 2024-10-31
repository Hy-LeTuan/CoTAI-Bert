import psutil
import time
from datetime import datetime


def log_memory_usage(interval=5, output_file="memory_log.txt"):
    """
    Logs system memory usage at regular intervals for a specified duration.

    Parameters:
    - interval: Time in seconds between measurements
    - duration: Total duration for logging in seconds
    - output_file: File path for the memory log output
    """

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(
            "Timestamp,Total Memory,Available Memory,Used Memory,Percent Used\n")

        while True:
            mem = psutil.virtual_memory()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"{timestamp},{round(mem.total)},{round(mem.available)},{round(mem.used)},{round(mem.percent)}\n"
            print(log_line)
            file.write(log_line)
            time.sleep(interval)


log_memory_usage(interval=2)
