from app.rag.retriever import search_document

results = search_document("tiktok 参数入参有几个？")
for index, item in enumerate(results):
    print(f"\n========== 结果 {index + 1} ==========")
    print("内容：", item["content"])
    print("来源：", item["source"])
    print("页码：", item["page"])
    print("距离分数：", item["score"])