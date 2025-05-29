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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import setproctitle

# Load environment variables from .env file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

# Set Hugging Face cache directory from environment variable
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/modeller/huggingface")

# Fetch embedding model name from environment or use default
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Set process title for easy identification in system monitors
setproctitle.setproctitle(f"EmbeddingModel-{EMBEDDING_MODEL_NAME}")

# Load embedding model dynamically
_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def embed_text(text: str) -> list:
    """
    Converts input text into a vector embedding.

    Args:
        text (str): The text string to be embedded.

    Returns:
        list: Embedding vector as a Python list of floats.
    """
    return _embedding_model.encode(text).tolist()

