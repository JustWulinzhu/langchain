import os
from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

DOCUMENT_DIR = os.getenv("DOCUMENT_DIR")

CHROMA_DIR = os.getenv("CHROMA_DIR")