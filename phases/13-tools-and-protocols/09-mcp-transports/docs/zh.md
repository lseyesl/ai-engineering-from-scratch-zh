# MCP 传输层

> MCP 传输层定义了客户端和服务器如何在底层进行通信。

**类型：** 构建（Build）
**语言：** TypeScript
**前置要求：** MCP 基础，TypeScript 基础
**时间：** ~45 分钟

## 学习目标

- 理解 MCP 传输层架构及其在协议栈中的角色
- 比较 stdio、SSE 和流式 HTTP 传输方式
- 使用不同的传输层实现一个 MCP 服务器
- 实现传输层抽象以支持多种传输方式
- 理解传输层级别的安全和序列化注意事项

## 什么是 MCP 传输层？

MCP 传输层提供了客户端和服务器之间的底层通信。它处理消息的序列化、传输和接收，而无需理解消息的内容。

```
┌─────────────────────────────────────────┐
│            MCP 协议层                    │
│   (请求、响应、通知、工具、资源、提示词)  │
├─────────────────────────────────────────┤
│            MCP 传输层                    │
│   (stdio、SSE、流式 HTTP)                │
├─────────────────────────────────────────┤
│         底层传输方式                      │
│   (标准输入/输出、HTTP、WebSocket)        │
└─────────────────────────────────────────┘
```

## 传输方式类型

MCP 定义了三种标准的传输方式：

### 1. stdio 传输

stdio 传输通过标准输入和标准输出流进行通信。它适用于本地进程间通信，是最简单的设置方式。

```typescript
// stdio 传输示例
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server(
  {
    name: "example-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// 使用 stdio 传输连接
const transport = new StdioServerTransport();
await server.connect(transport);
```

**特点：**
- 零配置——自动使用进程的标准流
- 最适合本地子进程
- 不支持远程连接
- 最简单的调试方式

### 2. SSE 传输

SSE（服务器发送事件）传输通过 HTTP 实现单向通信，服务器通过 SSE 向客户端推送事件，客户端通过单独的 POST 端点发送消息。

```typescript
// SSE 服务器端示例
import express from "express";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";

const app = express();

app.get("/sse", async (req, res) => {
  const transport = new SSEServerTransport("/messages", res);
  const server = createServer(); // 你的服务器实例
  await server.connect(transport);
});

app.post("/messages", async (req, res) => {
  const transport = SSEServerTransport.getTransport(req.body.sessionId);
  if (transport) {
    await transport.handleMessage(req.body);
  }
  res.status(200).json({ ok: true });
});
```

```typescript
// SSE 客户端端示例
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";

const client = new Client(
  {
    name: "example-client",
    version: "1.0.0",
  },
  {
    capabilities: {},
  }
);

const transport = new SSEClientTransport(
  new URL("http://localhost:3000/sse")
);
await client.connect(transport);
```

**特点：**
- 支持远程连接
- 单向服务器到客户端的事件推送
- 客户端通过 HTTP POST 发送消息
- 支持现有 HTTP 基础设施

### 3. 流式 HTTP 传输

流式 HTTP 传输提供了一种更现代的流式通信方式，支持请求和响应流的双向流式传输。

```typescript
// 流式 HTTP 传输示例
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";

// 服务器端
const transport = new StreamableHTTPServerTransport({
  endpoint: "/mcp",
});

// 处理传入连接
app.post("/mcp", async (req, res) => {
  await transport.handleRequest(req, res);
});
```

**特点：**
- 真正的双向流式传输
- 基于标准 HTTP
- 支持现有 HTTP 基础设施
- 适合服务器到服务器的通信

## 构建抽象传输层

让我们构建一个抽象层，支持在同一服务器上切换或组合多种传输方式：

```typescript
// transport-adapter.ts
import { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import { JSONRPCMessage } from "@modelcontextprotocol/sdk/types.js";

// 为消息处理定义接口
export interface TransportHandler {
  onMessage: (message: JSONRPCMessage) => Promise<void>;
  onError?: (error: Error) => void;
  onClose?: () => void;
}

// 传输适配器基类
export abstract class TransportAdapter implements Transport {
  protected handler?: TransportHandler;
  abstract start(): Promise<void>;
  abstract send(message: JSONRPCMessage): Promise<void>;
  abstract close(): Promise<void>;

  onclose?: () => void;
  onerror?: (error: Error) => void;
  onmessage?: (message: JSONRPCMessage) => void;
  sessionId?: string;

  async connect(handler: TransportHandler): Promise<void> {
    this.handler = handler;
    await this.start();
  }
}

// stdio 适配器
export class StdioAdapter extends TransportAdapter {
  async start(): Promise<void> {
    // stdio 初始化逻辑
  }

  async send(message: JSONRPCMessage): Promise<void> {
    process.stdout.write(JSON.stringify(message) + "\n");
  }

  async close(): Promise<void> {
    process.exit(0);
  }
}

// HTTP 适配器
export class HTTPAdapter extends TransportAdapter {
  private endpoint: string;

  constructor(endpoint: string) {
    super();
    this.endpoint = endpoint;
  }

  async start(): Promise<void> {
    // HTTP 服务器初始化
  }

  async send(message: JSONRPCMessage): Promise<void> {
    const response = await fetch(this.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message),
    });
    const data = await response.json();
    this.onmessage?.(data as JSONRPCMessage);
  }

  async close(): Promise<void> {
    // 清理逻辑
  }
}
```

## 传输层选择指南

| 传输方式 | 何时使用 | 限制 |
|---------|---------|------|
| **stdio** | 本地工具、CLI 集成、开发环境 | 仅限本地 |
| **SSE** | 远程客户端、Web 应用 | 单向服务器推送 |
| **流式 HTTP** | 服务器到服务器、流式响应 | 较新的标准，库支持较少 |

## 练习

1. **实现 stdio 回显服务器**：创建一个使用 stdio 传输的 MCP 服务器，回显收到的任何消息。

2. **添加 SSE 支持**：修改你的 MCP 服务器，同时支持 stdio 和 SSE 传输。

3. **健康检查端点**：在你的 MCP 服务器中添加一个暴露连接统计信息的健康检查端点。

4. **跨传输方式测试**：编写一个测试，验证你的服务器在使用不同传输方式时表现一致。

5. **自定义日志传输**：创建一个同时向标准输出和日志文件写入的传输适配器。

## 术语表

- **传输层**：协议栈中处理底层数据移动的层。
- **stdio**：通过标准输入/输出流进行通信。
- **SSE（服务器发送事件）**：一种允许服务器向客户端推送事件的标准。
- **序列化**：将数据结构转换为可传输格式的过程。
- **流式 HTTP**：一种支持双向流式传输的 HTTP 通信方式。

## 延伸阅读

- MCP 传输层规范
- MDN 上的 SSE 文档
- HTTP 流式传输标准
- Node.js 流式传输文档
