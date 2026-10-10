import uuid
from datetime import datetime

def generate_video_id():
    now = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    return f"{now}_{uuid.uuid4().hex[:8]}"
