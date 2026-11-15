from mcp.server import MCPServer

mcp_server = MCPServer("order_server")

# 模拟订单数据
ORDERS = {
    "order1": {
        "order_id": "001",
        "amount": 199,
        "status": "已发货"
    },
    "order2": {
        "order_id": "002",
        "amount": 899,
        "status": "已付款"
    }
}

# 模拟物流数据
LOGISTICS = {
    "order1": "已到达北京分拨中心",
    "order2": "暂无物流信息"
}

@mcp_server.tool()
def get_order_by_id(order_id: str) -> str :
    order = ORDERS.get(order_id) # 别用ORDERS[order_id] 如果某个key不存在会直接报错
    if order is None :
        return "订单不存在"
    # str 类型转换，转换成字符串类型
    return str(order)

@mcp_server.tool()
def get_logistics_by_order_id(order_id: str) -> str :
    order = ORDERS.get(order_id)
    if order is None:
        return "订单不存在"
    logistics = LOGISTICS.get(order_id)
    if logistics is None :
        return "物流信息不存在"
    return str(logistics)

@mcp_server.tool()
def get_amount_by_order_id(order_id: str) -> str :
    order = ORDERS.get(order_id)
    if order is None :
        return "订单不存在"
    return f"订单金额为：{order['amount']}"

if __name__ == "__main__":
    mcp_server.run()