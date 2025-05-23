from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv

load_dotenv()

qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST"),
    port=int(os.getenv("QDRANT_PORT"))
)

collection_name = os.getenv("QDRANT_COLLECTION", "chat_memory")

# Define schema for payload filtering and indexing
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=int(os.getenv("VECTOR_SIZE", 384)),
        distance=models.Distance.COSINE
    ),
    optimizers_config=models.OptimizersConfigDiff(
        default_segment_number=1
    ),
    # Optional but recommended for structured payload filtering
    on_disk_payload=True
)

# Optional: set up payload index for fast filtering (by agent, topic, etc.)
qdrant.set_payload_schema(
    collection_name=collection_name,
    schema={
        "role": models.PayloadSchemaType.keyword,
        "agent": models.PayloadSchemaType.keyword,
        "topics": models.PayloadSchemaType.keyword,
        "emotional_tone": models.PayloadSchemaType.keyword,
        "tool_used": models.PayloadSchemaType.keyword,
        "trust_level": models.PayloadSchemaType.float,
        "love_level": models.PayloadSchemaType.float,
        "humor_level": models.PayloadSchemaType.float,
        "anger_level": models.PayloadSchemaType.float,
        "joy_level": models.PayloadSchemaType.float,
        "sadness_level": models.PayloadSchemaType.float,
        "fear_level": models.PayloadSchemaType.float,
        "curiosity_level": models.PayloadSchemaType.float,
        "dominance_level": models.PayloadSchemaType.float,
        "submission_level": models.PayloadSchemaType.float,
    }
)

print(f"✅ Qdrant collection '{collection_name}' initialized with emotional metadata support.")
