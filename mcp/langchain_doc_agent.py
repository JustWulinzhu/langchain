import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

# 连接 LangChain 官方文档 MCP

llm = ChatOpenAI(
    model = "deepseek-flash",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("DEEPSEEK_BASE_URL")
)

async def main() :
    # 连接外部MCP server
    async with MCPAdapter(
        "https://docs.langchain.com/mcp"
    ) as adapter :
        tools = await adapter.list_tools()
        for tool in tools :
            print(f"工具名称：{tool.name}")

        agent = create_agent(
            model = llm,
            tools = tools,
            system_prompt = "你是langchain官方文档助手，回答问题时依据提供的工具回答，禁止编造数据"
        )

        result = await agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": "langchain是什么？"
                }
            ]
        })
        print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())