import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
import asyncio
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model = "deepseek-flash",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("DEEPSEEK_BASE_URL")
)

async def main() :
    config = {
        "order": {
            "command": "python",
            "args": [
                "mcp/order_server.py"
            ],
            "transport": "stdio"
        }
    }
    async with MCPAdapter(config) as adapter :
        tools = await adapter.list_tools()
        for tool in tools :
            print(tool.name)

        agent = create_agent(
            model = llm,
            tools = tools,
            system_prompt = """
            你是一个订单查询小助手，
            用户查询订单信息、物流信息等需要调用工具查询，
            禁止编造数据
            """
        )

        result = await agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": "帮我查询订单order1，告诉我订单详情，金额、物流等"
                }
            ]
        })

        print(result["messages"][-1].content)

if __name__ == "__main__":
    # 启动异步引擎，把 main() 函数的所有流程跑起来
    asyncio.run(main())