import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_dir=os.path.join(os.getcwd(),"logs")
os.makedirs(log_dir,exist_ok=True)
log_file_path=os.path.join(log_dir,LOG_FILE)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
)