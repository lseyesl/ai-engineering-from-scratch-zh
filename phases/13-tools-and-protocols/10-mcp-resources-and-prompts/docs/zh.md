# MCP 资源与提示词

> 资源向 LLM 提供结构化上下文，提示词为常见交互提供可复用的模板。

**类型：** 构建（Build）
**语言：** TypeScript
**前置要求：** MCP 基础，MCP 服务器基础
**时间：** ~50 分钟

## 学习目标

- 理解 MCP 中资源与提示词的角色
- 在 MCP 服务器中实现资源提供
- 创建可复用的提示词模板
- 将资源与提示词链接以实现动态上下文
- 构建一个带有资源和提示词的实际 MCP 服务器

## 什么是 MCP 资源？

MCP 资源是 LLM 可以读取的结构化数据块。它们提供了一种向 AI 模型暴露数据和上下文的方式，类似于 REST API 中的 GET 端点。

```
资源 URI 方案：
resource://{server}/{type}/{path}

示例：
docs://filesystem/readme.txt
db://users/active
api://weather/current?city=London
```

### 资源类型

```typescript
// 基本资源接口
interface Resource {
  uri: string;           // 资源的唯一标识符
  name: string;          // 人类可读的名称
  description?: string;  // 可选描述
  mimeType?: string;     // MIME 类型（默认 text/plain）
  text?: string;         // 文本内容
  blob?: string;         // 二进制内容（base64 编码）
}
```

## 什么是 MCP 提示词？

MCP 提示词是 LLM 交互的可复用的参数化模板。它们为常见操作封装了最佳实践和结构化的提示模式。

```typescript
// 基本提示词接口
interface Prompt {
  name: string;             // 提示词名称（用于引用）
  description?: string;     // 提示词用途描述
  arguments?: PromptArgument[];  // 参数定义
}

interface PromptArgument {
  name: string;             // 参数名称
  description?: string;     // 参数描述
  required?: boolean;       // 是否必需（默认 false）
}
```

## 构建启用资源的 MCP 服务器

让我们构建一个同时提供资源和提示词的 MCP 服务器：

```typescript
// resource-prompt-server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// 创建一个提供资源和提示词的 MCP 服务器
const server = new Server(
  {
    name: "knowledge-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      resources: {},  // 声明资源支持
      prompts: {},    // 声明提示词支持
    },
  }
);

// 1. 实现资源处理

// 资源列表
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: "docs://getting-started",
        name: "快速入门指南",
        description: "上手的基本步骤",
        mimeType: "text/markdown",
      },
      {
        uri: "docs://api-reference",
        name: "API 参考",
        description: "完整 API 文档",
        mimeType: "text/markdown",
      },
    ],
  };
});

// 资源读取
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const uri = request.params.uri;

  switch (uri) {
    case "docs://getting-started":
      return {
        contents: [
          {
            uri,
            mimeType: "text/markdown",
            text: `# 快速入门

## 安装
\`\`\`bash
npm install my-package
\`\`\`

## 使用方法
\`\`\`typescript
import { createApp } from 'my-package';
const app = createApp();
\`\`\``,
          },
        ],
      };
    case "docs://api-reference":
      return {
        contents: [
          {
            uri,
            mimeType: "text/markdown",
            text: `# API 参考

## createApp(options)
创建应用实例。

**参数：**
- \`options.name\`: string - 应用名称
- \`options.version\`: string - 版本号

**返回：** App 实例`,
          },
        ],
      };
    default:
      throw new Error(`资源未找到：${uri}`);
  }
});

// 2. 实现提示词处理

// 提示词列表
server.setRequestHandler(ListPromptsRequestSchema, async () => {
  return {
    prompts: [
      {
        name: "explain-code",
        description: "解释代码片段的功能",
        arguments: [
          {
            name: "code",
            description: "要解释的代码",
            required: true,
          },
          {
            name: "language",
            description: "编程语言",
            required: false,
          },
        ],
      },
      {
        name: "review-pr",
        description: "审查拉取请求的更改",
        arguments: [
          {
            name: "diff",
            description: "要审查的差异内容",
            required: true,
          },
          {
            name: "context",
            description: "额外上下文",
            required: false,
          },
        ],
      },
    ],
  };
});

// 提示词获取
server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "explain-code": {
      const code = args?.code || "";
      const language = args?.language || "unknown";
      return {
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: `请解释以下 ${language} 代码的功能：

\`\`\`${language}
${code}
\`\`\`

请包括：
1. 代码的总体目的
2. 关键部分及其作用
3. 任何值得注意的模式或习惯用法`,
            },
          },
        ],
      };
    }
    case "review-pr": {
      const diff = args?.diff || "";
      const context = args?.context || "无";
      return {
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: `请审查此拉取请求：

上下文：${context}

\`\`\`diff
${diff}
\`\`\`

请关注：
1. 正确性和潜在的错误
2. 代码质量和风格
3. 性能影响
4. 测试覆盖范围`,
            },
          },
        ],
      };
    }
    default:
      throw new Error(`提示词未找到：${name}`);
  }
});

// 启动服务器
const transport = new StdioServerTransport();
await server.connect(transport);
```

## 动态资源

资源也可以是动态的——基于查询参数或外部数据源返回不同内容：

```typescript
// 动态数据库查询资源
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const uri = new URL(request.params.uri);

  if (uri.protocol === "db:") {
    const table = uri.hostname;
    const id = uri.pathname.slice(1);

    // 模拟数据库查询
    const data = await queryDatabase(table, id);

    return {
      contents: [
        {
          uri: request.params.uri,
          mimeType: "application/json",
          text: JSON.stringify(data, null, 2),
        },
      ],
    };
  }

  throw new Error(`不支持协议：${uri.protocol}`);
});
```

## 将资源与提示词结合

结合资源和提示词创建强大的交互模式：

```typescript
// 使用资源填充提示词参数的提示词
server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  const { name } = request.params;

  if (name === "analyze-log") {
    // 从资源读取日志内容
    const logResource = await server.request(
      {
        method: "resources/read",
        params: { uri: "logs://latest" },
      },
      // @ts-ignore
      ReadResourceRequestSchema
    );

    return {
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: `分析以下最近的日志条目：

${logResource.contents[0].text}

识别：
1. 任何错误或警告
2. 异常模式
3. 建议的后续步骤`,
          },
        },
      ],
    };
  }
});
```

## 最佳实践

1. **有意义的 URI 方案**：使用一致的 URI 方案，使资源易于发现和浏览。
2. **适当的 MIME 类型**：始终指定 MIME 类型以帮助客户端处理内容。
3. **描述性提示词参数**：提示词参数应具有清晰的描述和默认值。
4. **错误处理**：优雅处理缺失的资源或无效参数。
5. **缓存考虑**：对于频繁访问的资源实现缓存头部。

## 练习

1. **文件系统资源服务器**：创建一个向 LLM 暴露本地文件系统目录的 MCP 服务器。

2. **代码审查提示词**：实现一个提示词，接受 Git 差异并为团队生成标准化的代码审查。

3. **动态天气 API 资源**：构建一个通过 API 获取实时天气数据并将其作为 MCP 资源暴露的服务器。

4. **带上下文注入的提示词**：创建一个自动从资源注入相关上下文的提示词。

5. **资源订阅**：实现当底层数据变化时通知客户端资源变化的资源订阅。

## 术语表

- **资源**：LLM 可以读取的结构化数据块。
- **提示词模板**：带参数的预定义 LLM 交互模板。
- **MIME 类型**：标识文件格式和内容的媒体类型。
- **URI 方案**：统一资源标识符的前缀部分，标识协议或命名空间。
- **动态资源**：基于请求参数或外部条件返回不同内容的资源。

## 延伸阅读

- MCP 资源规范
- MCP 提示词规范
- URL 和 URI 标准（RFC 3986）
- MIME 类型注册表
