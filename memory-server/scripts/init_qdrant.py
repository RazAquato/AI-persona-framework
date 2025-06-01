# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

import os
import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
EMOTION_PATH = os.path.join(BASE_DIR, "..", "config", "emotions.yaml")

load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))


# Connect to Qdrant
qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST"),
    port=int(os.getenv("QDRANT_PORT"))
)

collection_name = os.getenv("QDRANT_COLLECTION", "chat_memory")

# Load emotion definitions
with open(EMOTION_PATH, "r") as f:
    config = yaml.safe_load(f)

emotions = config.get("emotions", {})

# Base payload schema with standard fields
payload_schema = {
    "role": models.PayloadSchemaType.KEYWORD,
    "agent": models.PayloadSchemaType.KEYWORD,
    "topics": models.PayloadSchemaType.KEYWORD,
    "emotional_tone": models.PayloadSchemaType.KEYWORD,
    "tool_used": models.PayloadSchemaType.KEYWORD
}

# Dynamically add emotion fields based on YAML definition
for emotion_name, emotion_type in emotions.items():
    if emotion_type == "float":
        payload_schema[f"{emotion_name}_level"] = models.PayloadSchemaType.FLOAT
    elif emotion_type == "keyword":
        payload_schema[f"{emotion_name}_level"] = models.PayloadSchemaType.KEYWORD
    # Add support for more types if needed

# Create or reset Qdrant collection
# Check if collection exists
if qdrant.collection_exists(collection_name):
    print(f"Collection '{collection_name}' already exists. Deleting and recreating...")
    qdrant.delete_collection(collection_name)

# Now create the collection
qdrant.create_collection(
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

# Apply payload schema
#qdrant.set_payload_schema(
#    collection_name=collection_name,
#    schema=payload_schema
#)

print(f"Qdrant collection '{collection_name}' initialized with dynamic emotion schema from {EMOTION_PATH}.")
