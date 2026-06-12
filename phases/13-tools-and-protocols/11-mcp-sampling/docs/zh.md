# MCP 采样

> MCP 采样允许服务器通过客户端请求 LLM 补全，实现由服务器驱动的动态 AI 交互。

**类型：** 构建（Build）
**语言：** TypeScript
**前置要求：** MCP 基础，MCP 服务器基础
**时间：** ~40 分钟

## 学习目标

- 理解 MCP 中采样的概念及其目的
- 实现服务器端的采样请求
- 创建带采样支持的交互式工具
- 理解采样的速率限制和成本考虑
- 对比采样与直接 API 调用的优劣

## 什么是 MCP 采样？

MCP 采样允许服务器向客户端请求 LLM 补全。服务器不再需要直接调用 LLM API，而是发送一个包含消息和参数的采样请求，由客户端代为处理 LLM 调用。

```
┌─────────────┐     采样请求     ┌─────────────┐     LLM API     ┌───────────┐
│ MCP 服务器  │ ──────────────> │ MCP 客户端  │ ──────────────> │ LLM API   │
│             │ <────────────── │             │ <────────────── │ (OpenAI等)│
│             │   采样响应      │             │    完成响应     │           │
└─────────────┘                 └─────────────┘                 └───────────┘
```

### 为什么使用采样？

1. **简化服务器架构**：服务器不需要 API 密钥或直接调用 LLM API
2. **集中式成本管理**：客户端处理 LLM API 调用和密钥管理
3. **统一的速率限制**：客户端可以强制执行一致的速率限制和配额
4. **用户选择模型**：用户可以决定使用哪个 LLM 提供商/模型

## 采样请求流程

```typescript
// 采样请求（从服务器到客户端）
interface CreateSampleRequest {
  method: "sampling/createMessage";
  params: {
    messages: SamplingMessage[];    // 要发送给 LLM 的消息
    modelPreferences?: ModelPreferences; // 模型选择偏好
    systemPrompt?: string;           // 可选的系统提示词
    includeContext?: "none" | "thisServer" | "allServers"; // 上下文包含策略
    temperature?: number;            // 采样温度
    maxTokens?: number;              // 最大令牌数
    stopSequences?: string[];        // 停止序列
    metadata?: Record<string, unknown>; // 额外元数据
  };
}

// 采样消息
interface SamplingMessage {
  role: "user" | "assistant";
  content: TextContent | ImageContent;
}

// 文本内容
interface TextContent {
  type: "text";
  text: string;
}

// 图片内容
interface ImageContent {
  type: "image";
  data: string;         // base64 编码
  mimeType: string;     // 图片 MIME 类型
}

// 采样响应
interface CreateSampleResult {
  model: string;          // 使用的模型
  role: "assistant";
  content: TextContent | ImageContent;
  stopReason?: string;    // 停止原因
}
```

## 实现服务器端采样

让我们构建一个在工具处理程序中使用采样的 MCP 服务器：

```typescript
// sampling-server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
  CreateMessageRequestSchema,
  CreateMessageResult,
} from "@modelcontextprotocol/sdk/types.js";

const server = new Server(
  {
    name: "ai-assistant-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
      sampling: {},  // 声明采样支持
    },
  }
);

// 注册一个使用采样的工具
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "analyze_text") {
    const text = String(request.params.arguments?.text || "");

    // 请求客户端使用 LLM 分析文本
    const sampleResult = await server.request(
      {
        method: "sampling/createMessage",
        params: {
          messages: [
            {
              role: "user",
              content: {
                type: "text",
                text: `分析以下文本并提供总结：

${text}

请提供：
1. 主要观点
2. 关键主题
3. 总体语气`,
              },
            },
          ],
          includeContext: "none",
          maxTokens: 500,
          temperature: 0.3,
        },
      },
      CreateMessageRequestSchema
    );

    return {
      content: [
        {
          type: "text",
          text: `分析结果：

${sampleResult.content.text}

---
模型：${sampleResult.model}`,
        },
      ],
    };
  }

  throw new Error("未知工具");
});
```

## 模型偏好

服务器可以指定模型偏好，引导客户端选择合适的模型：

```typescript
// 模型偏好
interface ModelPreferences {
  hints?: ModelHint[];       // 模型选择提示
  costPriority?: number;     // 成本优先级（0-1，越高越优先）
  speedPriority?: number;    // 速度优先级（0-1，越高越优先）
  intelligencePriority?: number; // 智能优先级（0-1，越高越优先）
}

interface ModelHint {
  name?: string;              // 模型名称或前缀
}

// 示例：为不同任务指定偏好
const fastCheapPrefs: ModelPreferences = {
  hints: [{ name: "gpt-4o-mini" }],
  costPriority: 0.8,
  speedPriority: 0.9,
  intelligencePriority: 0.3,
};

const accuratePrefs: ModelPreferences = {
  hints: [{ name: "claude" }],
  costPriority: 0.3,
  speedPriority: 0.4,
  intelligencePriority: 0.9,
};
```

## 客户端实现采样支持

MCP 客户端需要处理 `sampling/createMessage` 请求：

```typescript
// 客户端采样处理程序
class MyClient {
  private llmProvider: LLMProvider;

  constructor(provider: LLMProvider) {
    this.llmProvider = provider;
  }

  // 处理传入的采样请求
  async handleSamplingRequest(
    request: CreateSampleRequest,
    serverInfo: { name: string; version: string }
  ): Promise<CreateSampleResult> {
    const { messages, systemPrompt, temperature, maxTokens } = request.params;

    // 添加系统提示词（如果提供）
    const fullMessages = systemPrompt
      ? [{ role: "system", content: systemPrompt }, ...messages]
      : messages;

    // 调用 LLM API
    const response = await this.llmProvider.complete({
      messages: fullMessages,
      temperature: temperature ?? 0.7,
      maxTokens: maxTokens ?? 1000,
      model: this.selectModel(request.params.modelPreferences),
    });

    return {
      model: response.model,
      role: "assistant",
      content: {
        type: "text",
        text: response.content,
      },
      stopReason: response.stopReason,
    };
  }

  private selectModel(prefs?: ModelPreferences): string {
    // 根据偏好实现模型选择
    if (!prefs || !prefs.hints?.length) {
      return "gpt-4o"; // 默认模型
    }

    if (prefs.speedPriority && prefs.speedPriority > 0.7) {
      return "gpt-4o-mini"; // 快速模式使用小模型
    }

    if (prefs.intelligencePriority && prefs.intelligencePriority > 0.7) {
      return "claude-3-opus"; // 高质量模式使用强模型
    }

    return "gpt-4o";
  }
}
```

## 上下文包含策略

`includeContext` 参数控制服务器在采样请求中包含多少上下文：

| 值 | 行为 | 使用场景 |
|-----|----------|-----------|
| `none` | 无额外上下文 | 独立请求 |
| `thisServer` | 包含此服务器的上下文 | 需要跨工具信息的服务器 |
| `allServers` | 包含所有服务器的上下文 | 需要全局上下文的服务器 |

## 采样最佳实践

1. **最小权限上下文**：除非必要，否则使用 `includeContext: "none"` 以减少令牌使用。

2. **设置限制**：始终设置 `maxTokens` 以防止无限生成。

3. **温度调整**：对分析任务使用较低温度（0.1-0.3），对创造性任务使用较高温度（0.7-0.9）。

4. **错误处理**：实现采样失败的优雅回退。

5. **成本意识**：对简单任务使用成本优先模型，对复杂分析使用智能优先模型。

## 练习

1. **文本摘要工具**：创建一个使用采样为长文本生成摘要的 MCP 工具。

2. **代码审查工具**：实现一个使用采样进行代码审查并提供改进建议的工具。

3. **多步骤分析**：构建一个使用多次采样调用的工具，级联结果以执行复杂分析。

4. **自定义模型选择器**：根据任务类型和优先级实现精细的模型选择逻辑。

5. **采样监控**：创建一个记录和监控所有采样请求以跟踪使用情况和成本的系统。

## 术语表

- **采样**：MCP 中通过客户端请求 LLM 补全的过程。
- **ModelPreferences**：用于引导模型选择的偏好设置。
- **上下文包含**：控制采样请求中包含多少上下文。
- **补全**：LLM API 生成的响应。
- **令牌**：LLM 使用的文本单位（单词或子词）。

## 延伸阅读

- MCP 采样规范
- OpenAI API 文档
- Anthropic API 文档
- LLM 成本和基准对比
