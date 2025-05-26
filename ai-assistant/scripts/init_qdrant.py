from qdrant_client import QdrantClient, models
import os
import yaml
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.expanduser("~/ai-assistant/config/.env"))

qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST"),
    port=int(os.getenv("QDRANT_PORT"))
)

collection_name = os.getenv("QDRANT_COLLECTION", "chat_memory")

# Load emotion definitions from YAML
with open(os.path.expanduser(os.getenv("EMOTION_CONFIG_PATH", "./config/emotions.yaml")), "r") as f:
    config = yaml.safe_load(f)

emotions = config.get("emotions", {})

# Base schema
payload_schema = {
    "role": models.PayloadSchemaType.keyword,
    "agent": models.PayloadSchemaType.keyword,
    "topics": models.PayloadSchemaType.keyword,
    "tool_used": models.PayloadSchemaType.keyword,
    "emotional_tone": models.PayloadSchemaType.keyword
}

# Add all emotion float fields dynamically
for emotion in emotions:
    payload_schema[f"{emotion}_level"] = models.PayloadSchemaType.float

# Create collection
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=int(os.getenv("VECTOR_SIZE", 384)),
        distance=models.Distance.COSINE
    ),
    optimizers_config=models.OptimizersConfigDiff(
        default_segment_number=1
    ),
    on_disk_payload=True
)

# Apply schema
qdrant.set_payload_schema(
    collection_name=collection_name,
    schema=payload_schema
)

print(f"✅ Qdrant collection '{collection_name}' initialized with dynamic emotion schema.")