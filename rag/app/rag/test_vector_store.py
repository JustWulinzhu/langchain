from app.loader.document_loader import load_pdf
from app.rag.splitter import document_splitter
from app.rag.vector_store import save_chunks

pdf_documents = load_pdf("rag/data/docs/TikTok_API_Get_Campaigns_v1.3.pdf")
chunks = document_splitter(pdf_documents)
ids = save_chunks(chunks)

print("知识库构建完成，chunks已写入到向量数据库")