from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv

load_dotenv()

qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST"),
    port=int(os.getenv("QDRANT_PORT"))
)

collection_name = os.getenv("QDRANT_COLLECTION", "chat_memory")

qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=int(os.getenv("VECTOR_SIZE", 384)),
        distance=models.Distance.COSINE
    ),
    optimizers_config=models.OptimizersConfigDiff(
        default_segment_number=1
    )
)

print(f"✅ Qdrant collection '{collection_name}' initialized.")
