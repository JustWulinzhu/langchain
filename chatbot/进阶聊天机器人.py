import os
# LangChain 专门适配 DeepSeek 开发的包
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
load_dotenv()

# 作用：从 LangChain 核心库导入三种标准对话消息类。由于大模型本身是“无状态”的（记不住上一句话），每次请求都必须把历史对话打包发给它，因此需要用这三个类来区分每句话的角色：
# SystemMessage：系统指令，设定 AI 的人设、规则和回复要求。
# HumanMessage：人类用户的输入/提问。
# AIMessage：AI 模型的历史回复。
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请在 .env 文件中配置 DEEPSEEK_API_KEY")

model = os.getenv("DEEPSEEK_MODEL")
if not model:
    raise ValueError("请在 .env 文件中配置 DEEPSEEK_MODEL")

# 创建deepseek大模型
llm = ChatDeepSeek(
    model = model,
    api_key = api_key,
    temperature = 0, # 控制生成结果的随机性，0 表示几乎不随机，每次输出最确定、最严谨的答案，适合编程和事实问答；数值越高回答越发散有创意
    timeout = 60, # 请求超时时间为 60 秒
    max_retries = 2 # 当遇到网络波动或服务端临时错误时，自动重试最多 2 次
)

# 定义系统提示词
SYSTEM_PROMPT = """
你是一个专业的 AI 助手。

要求：
1. 使用中文回答用户。
2. 回答尽量清晰、准确。
3. 编程问题尽量提供代码示例。
4. 如果不知道答案，不要编造。
5. 对于复杂问题，分步骤解释。
"""

# 定义ai角色和职责，将纯文本的系统提示词包装成 LangChain 认识的 SystemMessage 对象
# 给ai的提示词。无论后面聊了多少轮，这条指令通常都会放在每次请求列表的最前面，用来约束 AI 的身份和回复语言。
system_message = SystemMessage(
    content = SYSTEM_PROMPT
)

# 创建聊天历史存储变量
# 初始化一个空列表，充当机器人的“短期记忆库”。程序运行期间，所有的用户提问和 AI 回答都会按时间顺序存进这个列表里
chat_history = []

# 定义处理单轮对话的函数，接收用户的输入文本 user_input 作为参数
def chat(user_input):
    # 声明函数内部使用的 chat_history 是外部的全局变量（因为函数末尾有 chat_history = chat_history[-20:] 的重新赋值操作，
    # 如果不加 global，Python 会将其视为局部变量从而报错）
    global chat_history

    # 组装给AI的完整消息
    # 作用：组装本次要发送给大模型的完整上下文列表：
    # 第一条永远是 system_message，确保给ai的提示词
    # *chat_history 使用解包操作符 *，将历史记录列表中的所有 HumanMessage 和 AIMessage 依次展开铺平放进新列表中。
    # 最后一条是把当前用户刚输入的内容包装成 HumanMessage。
    messages = [
        system_message,
        *chat_history,
        HumanMessage(content = user_input)
    ]

    try:
        # end = ""：取消 print 默认的换行，让后续 AI 的回复紧跟在冒号后面
        # flush = True：强制刷新输出缓冲区，确保文字立刻显示在屏幕上，而不是等遇到换行符才一起输出（实现打字机效果的关键）
        print("\nAI: ", end = "", flush = True)
        # 初始化一个空字符串，用来把流式输出的碎片拼接成完整的 AI 回答，以便后续存入历史记录。
        answer = ""
        # 流式调用，区别于一次性返回全部结果的 .invoke()，模型每生成一个词（Token）就会推送过来一块
        response = llm.stream(messages)
        # 遍历流式返回的每一个数据块 chunk
        for chunk in response:
            content = chunk.content
            # 不换行且实时刷新地打印到终端
            print(content, end = "", flush = True)
            answer += content

        # 空行
        print()

        # 作用：本轮对话成功完成后，将本轮的“用户提问”和拼接完整的“AI 回答”依次追加到 chat_history 列表中，形成记忆。
        # 保存用户消息
        chat_history.append(
            HumanMessage(content = user_input)
        )
        # 保存AI的回答
        chat_history.append(
            AIMessage(content = answer)
        )

        # 只保留最近 20 条消息
        # 作用：使用列表切片 [-20:] 截取列表最后 20 个元素（即最近 10 轮一问一答）。
        # 这是为了防止对话过长导致超出模型的上下文窗口限制（Token 上限），同时节省 API 调用费用
        chat_history = chat_history[-20:]
    except Exception as e:
        print(f"系统异常: {e}")

# 定义主函数，负责终端界面初始化和接收用户循环输入。
def main():
    # 默认提示
    print("=" * 50)
    print("欢迎使用 AI 对话机器人")
    print("模型：", model)
    print("输入 exit 退出")
    print("输入 /clear 清空聊天记录")
    print("=" * 50)

    while True:
        # 作用：阻塞程序运行，在终端显示 "用户：" 并等待用户输入内容，按下回车后将输入的字符串赋值给 user_input。
        user_input = input("用户：")
        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "退出"]:
            print("AI：再见！")
            break

        if user_input == "/clear":
            chat_history.clear()
            print("AI：聊天记录已清空。")
            continue

        chat(user_input)

if __name__ == "__main__":
    main()