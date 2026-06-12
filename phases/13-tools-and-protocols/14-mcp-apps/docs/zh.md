# MCP 应用

> MCP 应用展示了如何将 MCP 集成到实际应用中，从聊天界面到全功能的 AI 驱动开发环境。

**类型：** 构建（Build）
**语言：** TypeScript
**前置要求：** MCP 基础，MCP 客户端开发
**时间：** ~60 分钟

## 学习目标

- 理解 MCP 应用的架构
- 构建一个与 MCP 服务器通信的聊天界面
- 实现工具执行的 UI 组件
- 创建资源浏览器和提示词选择器
- 设计一个完整的 MCP 客户端应用

## MCP 应用架构

MCP 应用由几个层组成，共同实现与 AI 模型的集成交互：

```
┌─────────────────────────────────────┐
│           用户界面层                  │
│  ┌─────────┐ ┌────────┐ ┌────────┐  │
│  │ 聊天界面 │ │工具UI  │ │资源    │  │
│  │         │ │组件    │ │浏览器  │  │
│  └─────────┘ └────────┘ └────────┘  │
├─────────────────────────────────────┤
│          应用逻辑层                   │
│  ┌─────────┐ ┌────────┐ ┌────────┐  │
│  │ 会话    │ │工具    │ │资源    │  │
│  │ 管理    │ │协调器  │ │管理器  │  │
│  └─────────┘ └────────┘ └────────┘  │
├─────────────────────────────────────┤
│          MCP 客户端层                │
│  ┌──────────────────────────────┐   │
│  │        MCP 客户端 SDK         │   │
│  └──────────────────────────────┘   │
├─────────────────────────────────────┤
│           传输层                     │
│     stdio    SSE    流式 HTTP        │
└─────────────────────────────────────┘
```

## 构建控制台聊天客户端

让我们从构建一个命令行 MCP 聊天应用开始：

```typescript
// mcp-chat-app.ts
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { spawn } from "child_process";
import * as readline from "readline";

interface MCPServerConfig {
  command: string;
  args: string[];
  env?: Record<string, string>;
}

class MCPChatApp {
  private client: Client;
  private serverConfig: MCPServerConfig;
  private transport: StdioClientTransport | null = null;
  private tools: any[] = [];
  private rl: readline.Interface;

  constructor(serverConfig: MCPServerConfig) {
    this.client = new Client(
      {
        name: "mcp-chat-app",
        version: "1.0.0",
      },
      {
        capabilities: {},
      }
    );

    this.serverConfig = serverConfig;
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
  }

  async start() {
    console.log("正在启动 MCP 聊天应用...\n");

    // 连接到 MCP 服务器
    this.transport = new StdioClientTransport({
      command: this.serverConfig.command,
      args: this.serverConfig.args,
    });

    await this.client.connect(this.transport);

    // 发现可用工具
    const result = await this.client.request(
      { method: "tools/list" },
      { schema: {} }
    );
    this.tools = (result as any).tools || [];

    this.printAvailableTools();
    this.startChatLoop();
  }

  private printAvailableTools() {
    console.log("可用工具：");
    for (const tool of this.tools) {
      console.log(`  📦 ${tool.name}: ${tool.description}`);
    }
    console.log("\n输入 'exit' 退出\n");
  }

  private startChatLoop() {
    this.rl.question("> ", async (input) => {
      if (input.toLowerCase() === "exit") {
        await this.shutdown();
        return;
      }

      await this.handleUserInput(input);
      this.startChatLoop();
    });
  }

  private async handleUserInput(input: string) {
    try {
      // 检测工具调用
      const toolMatch = input.match(/^@(\w+)\s+(.+)/);
      if (toolMatch) {
        const [, toolName, toolArgs] = toolMatch;
        await this.executeTool(toolName, toolArgs);
      } else {
        // 在消息中搜索工具相关意图
        const relevantTools = this.findRelevantTools(input);
        if (relevantTools.length > 0) {
          console.log("\n检测到相关工具：");
          for (const tool of relevantTools) {
            console.log(`  → @${tool.name} - ${tool.description}`);
          }
          console.log(`  使用 @工具名 参数 来调用工具\n`);
        } else {
          console.log("无相关工具。请表述得更具体一些。\n");
        }
      }
    } catch (error) {
      console.error("错误：", error);
    }
  }

  private findRelevantTools(input: string) {
    const keywords = input.toLowerCase().split(" ");
    return this.tools.filter((tool) => {
      const toolText = `${tool.name} ${tool.description}`.toLowerCase();
      return keywords.some((kw) => toolText.includes(kw));
    });
  }

  private async executeTool(toolName: string, args: string) {
    try {
      // 解析参数（简单 JSON 解析或键值对）
      let parsedArgs: Record<string, unknown>;
      try {
        parsedArgs = JSON.parse(args);
      } catch {
        parsedArgs = { text: args };
      }

      console.log(`\n正在执行 ${toolName}...\n`);

      const result = await this.client.request(
        {
          method: "tools/call",
          params: {
            name: toolName,
            arguments: parsedArgs,
          },
        },
        { schema: {} }
      );

      const response = result as any;
      if (response.content) {
        for (const item of response.content) {
          if (item.type === "text") {
            console.log(item.text);
          }
        }
      }
      console.log(); // 空行
    } catch (error) {
      console.error("工具执行失败：", error);
    }
  }

  async shutdown() {
    this.rl.close();
    await this.client.close();
    console.log("\n再见！");
    process.exit(0);
  }
}

// 使用方式
const app = new MCPChatApp({
  command: "node",
  args: ["path/to/mcp-server/index.js"],
});

app.start().catch(console.error);
```

## 构建 Web MCP 界面

现在让我们用 React 构建一个基于 Web 的 MCP 客户端：

```typescript
// MCPSessionManager.ts - Web 应用的 MCP 客户端层
import { Client } from "@modelcontextprotocol/sdk/client/index.js";

export interface ToolInfo {
  name: string;
  description: string;
  inputSchema: object;
}

export class MCPSessionManager {
  private client: Client;
  private tools: ToolInfo[] = [];
  private connected = false;

  constructor() {
    this.client = new Client(
      {
        name: "mcp-web-app",
        version: "1.0.0",
      },
      {
        capabilities: {},
      }
    );
  }

  async connect(transportUrl: string) {
    // 使用 SSE 或流式 HTTP
    const { SSEClientTransport } = await import(
      "@modelcontextprotocol/sdk/client/sse.js"
    );

    const transport = new SSEClientTransport(new URL(transportUrl));
    await this.client.connect(transport);

    await this.refreshTools();
    this.connected = true;
  }

  async refreshTools() {
    const result = await this.client.request(
      { method: "tools/list" },
      { schema: {} }
    );
    this.tools = ((result as any).tools || []).map((t: any) => ({
      name: t.name,
      description: t.description,
      inputSchema: t.inputSchema,
    }));
  }

  async executeTool(name: string, args: object) {
    const result = await this.client.request(
      {
        method: "tools/call",
        params: { name, arguments: args },
      },
      { schema: {} }
    );
    return result;
  }

  getAvailableTools(): ToolInfo[] {
    return this.tools;
  }

  isConnected(): boolean {
    return this.connected;
  }

  async disconnect() {
    await this.client.close();
    this.connected = false;
  }
}
```

## React UI 组件

```typescript
// ToolPanel.tsx - 工具执行面板
import React, { useState } from "react";

interface ToolPanelProps {
  tools: ToolInfo[];
  onExecute: (name: string, args: object) => Promise<void>;
}

export function ToolPanel({ tools, onExecute }: ToolPanelProps) {
  const [selectedTool, setSelectedTool] = useState<string>("");
  const [args, setArgs] = useState<string>("{}");
  const [loading, setLoading] = useState(false);

  const handleExecute = async () => {
    setLoading(true);
    try {
      const parsed = JSON.parse(args);
      await onExecute(selectedTool, parsed);
    } catch (e) {
      console.error("执行失败：", e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tool-panel">
      <h3>工具</h3>
      <select
        value={selectedTool}
        onChange={(e) => setSelectedTool(e.target.value)}
      >
        <option value="">选择工具...</option>
        {tools.map((t) => (
          <option key={t.name} value={t.name}>
            {t.name} - {t.description}
          </option>
        ))}
      </select>

      {selectedTool && (
        <>
          <textarea
            value={args}
            onChange={(e) => setArgs(e.target.value)}
            rows={6}
            placeholder='{"key": "value"}'
          />
          <button onClick={handleExecute} disabled={loading}>
            {loading ? "执行中..." : "执行"}
          </button>
        </>
      )}
    </div>
  );
}

// ResourceBrowser.tsx - 资源浏览面板
interface Resource {
  uri: string;
  name: string;
  description?: string;
}

export function ResourceBrowser() {
  const [resources, setResources] = useState<Resource[]>([]);

  return (
    <div className="resource-browser">
      <h3>资源</h3>
      <div className="resource-list">
        {resources.map((r) => (
          <div key={r.uri} className="resource-item">
            <span className="resource-icon">📄</span>
            <div>
              <strong>{r.name}</strong>
              <p>{r.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

## 完整的 MCP 桌面应用架构

```typescript
// AppState.ts - 整体应用状态管理
interface AppState {
  server: MCPSessionManager;
  chat: ChatHistory;
  toolExecution: ToolState;
  resources: ResourceState;
  prompts: PromptState;
}

interface ChatHistory {
  messages: ChatMessage[];
  currentSession: string;
}

interface ChatMessage {
  role: "user" | "assistant" | "tool";
  content: string;
  timestamp: Date;
  toolCalls?: ToolCall[];
}

interface ToolCall {
  name: string;
  args: object;
  result: unknown;
  status: "pending" | "success" | "error";
}

// 主应用组件
export function MCPApp() {
  const [state, setState] = useState<AppState>(initialState);
  const [isConnected, setIsConnected] = useState(false);

  const connectToServer = async (url: string) => {
    const manager = new MCPSessionManager();
    await manager.connect(url);
    setIsConnected(true);
    setState((prev) => ({
      ...prev,
      server: manager,
    }));
  };

  // 工具执行与聊天集成
  const handleSendMessage = async (message: string) => {
    // 添加用户消息
    addChatMessage("user", message);

    // 检测工具意图
    const tools = state.server.getAvailableTools();
    const relevantTool = findBestTool(message, tools);

    if (relevantTool) {
      const result = await state.server.executeTool(
        relevantTool.name,
        extractArgs(message, relevantTool.inputSchema)
      );

      addChatMessage("tool", formatToolResult(result), {
        name: relevantTool.name,
        args: {},
        result,
        status: "success",
      });
    }
  };

  return (
    <div className="mcp-app">
      <ConnectionBar onConnect={connectToServer} isConnected={isConnected} />

      <div className="main-layout">
        <Sidebar>
          <ToolPanel
            tools={state.server.getAvailableTools()}
            onExecute={handleToolExecution}
          />
          <ResourceBrowser />
          <PromptSelector prompts={state.prompts.available} />
        </Sidebar>

        <MainContent>
          <ChatView
            messages={state.chat.messages}
            onSendMessage={handleSendMessage}
          />
          <ToolResultsPanel
            executions={state.toolExecution.history}
          />
        </MainContent>
      </div>
    </div>
  );
}
```

## 最佳实践

1. **传输选择**：对本地开发使用 stdio，对 Web 应用使用 SSE/流式 HTTP。
2. **工具发现**：启动时缓存工具列表，定期刷新。
3. **乐观 UI**：在等待工具完成时立即显示反馈。
4. **并发控制**：实现工具执行队列以防止过载。
5. **错误恢复**：工具失败时提供自动重试和回退选项。

## 练习

1. **聊天 + 文件系统**：构建一个连接到文件系统 MCP 服务器的聊天应用。

2. **多服务器仪表盘**：创建同时连接到多个 MCP 服务器并显示其所有工具和资源的 Web 仪表盘。

3. **工具执行链**：实现一个允许用户将多个工具调用串联成一个工作流的 UI。

4. **MCP 应用模板**：设计一个供其他开发者创建自己的 MCP 应用的模板。

5. **实时进度**：向 MCP 应用添加进度条和状态指示器。

## 术语表

- **传输适配器**：连接特定传输类型（stdio、SSE 等）的组件。
- **工具面板**：UI 组件，显示可用工具并允许执行。
- **资源浏览器**：UI 组件，浏览和读取 MCP 资源。
- **提示词选择器**：UI 组件，选择和自定义 MCP 提示词。
- **会话管理**：在应用会话期间管理 MCP 连接状态。

## 延伸阅读

- MCP 客户端 SDK 文档
- React 状态管理
- 桌面应用架构
- WebSocket 实时更新
