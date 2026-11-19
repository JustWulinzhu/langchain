from app.loader.document_loader import load_pdf
from app.rag.splitter import document_splitter

document = load_pdf("/Users/wulinzhu/Documents/hc/project/python/langchain/rag/data/docs/TikTok_API_Get_Campaigns_v1.3.pdf")
chunks = document_splitter(document)