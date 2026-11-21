from langchain_community.embeddings import ZhipuAIEmbeddings
from app.config import ZHIPU_EMBEDDING_MODEL, ZHIPU_API_KEY

def get_embedding_model():
    """
    创建 Embedding 模型（智谱 embedding-3）
    """
    embedding = ZhipuAIEmbeddings(
        model = ZHIPU_EMBEDDING_MODEL,
        api_key = ZHIPU_API_KEY,
    )
    return embedding


if __name__ == "__main__":
    model = get_embedding_model()
    # 作用：真正发起网络请求！
    # embed_query() 是 LangChain 规定的标准方法，专门用来把单条查询文本（比如用户提出的一个问题）转换成数字列表（即“向量”）
    # 智谱服务器收到 "测试文本" 后，会计算出它对应的语义特征，返回一个包含 2048 个浮点数（如 [0.014, -0.038, ...]）的Python list，存进变量 vec 中
    # 这里还有 embed_documents 方法，是用来把多个文档转成向量的
    vec = model.embed_query("测试文本")
    # 作用：计算列表 vec 的长度并打印。对于智谱 embedding-3，这里会输出 维度: 2048
    # 为什么要看维度？ 因为后面你在创建向量数据库（如 Milvus、PGVector、Chroma）的表结构时，必须提前声明向量字段的维度，两边的数字必须一模一样（比如都是 2048），否则入库会直接报错。
    print("维度:", len(vec))
    # 作用：使用 Python 的列表切片语法 [:10]，只取出并打印前 10 个数字。因为完整的向量有 2048 个数字，全部打印出来会刷屏，取前 10 个足以肉眼确认接口调用成功、数据格式正常。
    print("向量内容：", vec[:10])