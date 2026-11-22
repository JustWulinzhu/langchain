from langchain_openai import ChatOpenAI
from app.config import DEEPSEEK_API_KEY
from app.config import DEEPSEEK_BASE_URL
from app.config import DEEPSEEK_MODEL
from app.rag.retriever import search_document
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(
    model = DEEPSEEK_MODEL,
    base_url = DEEPSEEK_BASE_URL,
    api_key = DEEPSEEK_API_KEY,
    temperature = 0.1
)

def ask_question(question: str):
    """
    RAG问答：
    1、检索相关文档
    2、组装上下文
    3、调用 Deepseek
    4、返回答案和来源
    """
    # 第一步：检索相关文档
    docs = search_document(question, top_k = 5)
    if not docs:
        return {
            "answer": "知识库中没有检索到相关的文档，暂时无法回答！",
            "sources": []
        }

    # 第二步：组装上下文
    # enumerate的含义： 说白就是加了个索引下标
    # 假设 docs 列表里有 3 条检索出来的文档：docs = [文档A, 文档B, 文档C]
    # enumerate(docs)（英文意为“枚举/列举”）：
    # 它会把原来的列表包装成一组组 (序号, 元素) 的配对：
    # 第 1 组：(0, 文档A)
    # 第 2 组：(1, 文档B)
    # 第 3 组：(2, 文档C)
    #
    # for index, doc in ...：
    # 每次循环时，把上面配对里的两个值分别拆出来赋值给两个变量：
    # 第 1 轮循环：index = 0，doc = 文档A
    # 第 2 轮循环：index = 1，doc = 文档B
    # 第 3 轮循环：index = 2，doc = 文档C
    context_parts = []
    for index, doc in enumerate(docs):
        context_parts.append(
            f"""
            [资料 {index + 1}]
    来源：{doc['source']}
    页码：{doc['page']}
    正文：
    {doc['content']}
            """
        )
    # 用“两个换行符（即一个空行）”作为，把列表里的所有元素连接成一个大字符串。
    # 最终的context 可以作为完整的从文档中检索出来的内容
    context = "\n\n".join(context_parts)

    # 第三步：构建prompt
    # 系统提示词
    system_prompt = """
    你是一个专业的广告平台 API 技术文档助手。
    你的任务是根据提供的 API 文档回答用户问题。
    必须遵守以下规则：
    1. 优先依据提供的文档回答。
    2. 不允许编造接口、字段、参数和枚举值。
    3. 如果文档没有提供相关信息，明确说明文档中没有找到。
    4. 如果不同版本的文档存在差异，必须指出版本差异。
    5. 不要把推测当成官方结论。
    6. 回答尽量使用中文。
    7. 回答 API 字段时，尽量列出字段名称、类型、含义。
    8. 回答枚举值时，尽量使用表格。
    9. 在答案中标明参考资料编号。
    """
    # 用户提示词
    user_prompt = f"""
    以下是从知识库检索出来的 API 文档：
    {context}
    
    用户问题：
    {question}
    请根据文档回答，并标明引用的资料编号。
    """
    # 第四步：使用langchain封装的方法 调用deepseek大模型
    messages = [
        SystemMessage(content = system_prompt),
        HumanMessage(content = user_prompt)
    ]
    response = llm.invoke(messages)
    answer = response.content

    # 第五步：整理来源
    sources = []
    for index, doc in enumerate(docs):
        sources.append({
            "reference": index + 1,
            "source": doc["source"],
            "page": doc["page"],
            "content": doc["content"]
        })

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    # 1. 准备一个与你导入的 TikTok API 文档相关的测试问题
    question = "获取广告系列（Campaign）接口的请求地址是什么？有哪些必填参数？"
    # 2. 调用问答函数
    result = ask_question(question)
    # 3. 打印大模型生成的回答
    print("\n【AI 回答】：")
    print(result["answer"])

    # 4. 打印参考来源核对溯源信息
    print("\n" + "=" * 50 + "\n【参考来源】：")
    for src in result["sources"]:
        print(f"[资料 {src['reference']}] 文件: {src['source']} | 页码: {src['page']}")
        # 只打印前100个字符预览，避免刷屏
        print(f"内容预览: {src['content'][:100]}...\n")