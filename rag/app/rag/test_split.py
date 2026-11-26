from app.loader.loader_to_markdown import parse_to_markdown
from app.rag.splitter import document_splitter,markdown_splitter

document = parse_to_markdown("/Users/ext.wulinzhu3/Documents/project/python/langchain/rag/data/docs/TikTok_API_Get_Campaigns_v1.3.pdf")
chunks = markdown_splitter(document)
print(chunks)