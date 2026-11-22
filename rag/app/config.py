import os
from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_EMBEDDING_MODEL = os.getenv("ZHIPU_EMBEDDING_MODEL")

DOCUMENT_DIR = os.getenv("DOCUMENT_DIR")
CHROMA_DIR = os.getenv("CHROMA_DIR")