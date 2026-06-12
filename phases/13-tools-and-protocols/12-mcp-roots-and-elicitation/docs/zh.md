# MCP 根与引导

> MCP 根定义了服务器的协作边界，引导通过迭代提示词优化来完善用户需求。

**类型：** 构建（Build）
**语言：** TypeScript
**前置要求：** MCP 基础，MCP 资源基础
**时间：** ~40 分钟

## 学习目标

- 理解 MCP 根及其在定义服务器范围中的角色
- 实现多个根以控制服务器上下文
- 理解引导（Elicitation）的概念及其在提示词优化中的角色
- 实现一个引导问题系统来完善用户输入
- 结合根和引导以创建更安全的 AI 交互

## 什么是 MCP 根？

MCP 根是客户端建议服务器关注的文件系统或资源目录。它们定义了服务器可以操作的边界，类似于容器中的挂载点。

```
根定义示例：

home://projects/my-app/           → 项目源文件
database://users/                  → 用户数据表
api://stripe/connected/           → Stripe API 集成
docs://company/policies/          → 公司文档
```

### 根类型

| 根类型 | 示例 URI | 目的 |
|----------|-------|-------------|
| 文件系统 | `file:///home/user/project/` | 目录访问 |
| 数据库 | `db://users/profiles/` | 数据库表 |
| API | `api://github/repos/` | API 端点 |
| 文档 | `docs://internal/wiki/` | 文档空间 |

## 根如何工作

当客户端连接时，它可以为服务器提供一个根列表，定义其工作范围：

```typescript
// MCP 初始化中的根建议
interface ClientCapabilities {
  roots?: {
    listChanged?: boolean;  // 服务器是否应监听根变化
  };
}

// 服务器请求根
interface ListRootsRequest {
  method: "roots/list";
}

// 根
interface Root {
  uri: string;        // 根 URI
  name?: string;      // 人类可读的名称
}
```

## 实现根感知的服务器

让我们构建一个使用根来定义其范围的 MCP 服务器：

```typescript
// roots-aware-server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListRootsRequestSchema,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

interface AppConfig {
  allowedPaths: string[];
  supportedDatabases: string[];
  apiEndpoints: string[];
}

class RootsAwareServer {
  private server: Server;
  private config: AppConfig = {
    allowedPaths: [],
    supportedDatabases: [],
    apiEndpoints: [],
  };

  constructor() {
    this.server = new Server(
      {
        name: "roots-aware-server",
        version: "1.0.0",
      },
      {
        capabilities: {
          tools: {},
          resources: {},
        },
      }
    );

    this.setupHandlers();
  }

  private setupHandlers() {
    // 从客户端获取根
    this.server.setRequestHandler(
      ListRootsRequestSchema,
      async () => {
        return {
          roots: [
            {
              uri: "file:///home/user/projects",
              name: "项目目录",
            },
            {
              uri: "db://analytics",
              name: "分析数据库",
            },
          ],
        };
      }
    );

    // 列出根范围内的资源
    this.server.setRequestHandler(
      ListResourcesRequestSchema,
      async () => {
        const resources = [];

        // 使用根来确定要暴露哪些资源
        for (const root of this.config.roots) {
          if (root.uri.startsWith("file://")) {
            resources.push({
              uri: `${root.uri}/readme.md`,
              name: "项目 README",
              mimeType: "text/markdown",
            });
          } else if (root.uri.startsWith("db://")) {
            resources.push({
              uri: `${root.uri}/schema`,
              name: "数据库模式",
              mimeType: "application/json",
            });
          }
        }

        return { resources };
      }
    );
  }

  // 使用根解析访问路径
  private resolvePath(requestedPath: string): string | null {
    // 检查请求的路径是否在配置的根范围内
    for (const root of this.config.roots) {
      if (requestedPath.startsWith(root.uri)) {
        return requestedPath;
      }
    }
    return null; // 路径超出范围
  }

  // 处理来自客户端的根更新
  updateRoots(roots: Root[]) {
    this.config.roots = roots;
    console.error(`根已更新：${roots.length} 个目录`);
  }

  async connect(transport: StdioServerTransport) {
    await this.server.connect(transport);
  }
}
```

## 什么是引导？

引导（Elicitation）是 MCP 服务器用来通过提问来完善和澄清用户请求的过程。它不是接受一个模糊的请求，而是主动引导用户提供更精确的规范。

```
没有引导：                         有引导：
用户："帮帮我"                     用户："帮帮我"
  ↓                                    ↓
服务器：执行默认操作               服务器："你想在什么方面获得帮助？
                                      1. 代码审查
                                      2. 调试问题  
                                      3. 架构建议
                                      4. 其他"
                                        ↓
                                   用户："代码审查"  → 更精确的操作
```

### 引导技术

1. **澄清性问题**：当请求不明确时提出问题
2. **选项选择**：提供结构化选项供选择
3. **逐步细化**：每次互动逐步缩小范围
4. **约束确认**：确认限制和边界

## 实现引导系统

让我们在我们的 MCP 服务器中添加引导支持：

```typescript
// elicitation-server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { CreateMessageRequestSchema } from "@modelcontextprotocol/sdk/types.js";

class ElicitationServer {
  private server: Server;

  constructor() {
    this.server = new Server(
      {
        name: "elicitation-server",
        version: "1.0.0",
      },
      {
        capabilities: {
          tools: {},
          sampling: {},  // 使用采样进行智能引导
        },
      }
    );

    this.setupElicitation();
  }

  private setupElicitation() {
    // 注册一个使用引导的工具
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      if (request.params.name === "generate_code") {
        const description = String(request.params.arguments?.description || "");

        if (this.needsClarification(description)) {
          // 需要澄清时发起引导
          const clarification = await this.elicitRequirements(description);

          return {
            content: [
              {
                type: "text",
                text: `您的请求需要更多细节：

${clarification.questions}

请重新运行此工具并提供更详细的描述。`,
              },
              {
                type: "resource",
                resource: {
                  text: JSON.stringify(clarification.suggestedTemplate, null, 2),
                  uri: "template://code-generation",
                  mimeType: "application/json",
                },
              },
            ],
          };
        }

        // 足够清晰时继续代码生成
        return this.generateCode(description);
      }

      throw new Error("未知工具");
    });
  }

  private needsClarification(input: string): boolean {
    const vaguePatterns = [
      /^(帮我|做一个|写一个|创建|make|create|write)/i,
      /^.{0,20}$/, // 非常短的输入
      /something|thing|stuff/i,
    ];

    return vaguePatterns.some(pattern => pattern.test(input));
  }

  private async elicitRequirements(
    description: string
  ): Promise<{
    questions: string;
    suggestedTemplate: Record<string, unknown>;
  }> {
    // 使用采样生成智能引导问题
    const sampleResult = await this.server.request(
      {
        method: "sampling/createMessage",
        params: {
          messages: [
            {
              role: "user",
              content: {
                type: "text",
                text: `用户请求："${description}"

这个请求太模糊了。生成 3 个澄清性问题来精确定义：
1. 代码的目的
2. 输入/输出格式
3. 任何特定约束

同时提供一个 JSON 模板供用户填写以规范他们的请求。`,
              },
            },
          ],
          maxTokens: 500,
          temperature: 0.7,
        },
      },
      CreateMessageRequestSchema
    );

    return {
      questions: sampleResult.content.text,
      suggestedTemplate: {
        purpose: "",
        inputs: [],
        outputs: [],
        constraints: [],
        language: "",
      },
    };
  }
}
```

## 结合根与引导

根和引导共同创造了更安全的 AI 交互：

```typescript
// 根约束生成引导问题
private async elicitWithRootContext(
  description: string
): Promise<string> {
  const rootContext = this.getRootContext();

  // 添加上下文感知的引导
  return `
在以下范围内：${rootContext}
请求：${description}

请提供：
1. 该范围内相关的具体文件或资源
2. 任何范围特定的安全考虑
3. 操作所需的访问级别
`;
}

private getRootContext(): string {
  return this.config.roots
    .map(r => `- ${r.name}: ${r.uri}`)
    .join("\n");
}
```

## 最佳实践

1. **根验证**：始终验证所有操作是否在配置的根范围内。
2. **引导渐进**：从广泛的问题开始，逐步缩小范围。
3. **根变更**：监听根变更并相应调整服务器行为。
4. **安全边界**：使用根作为安全边界——绝不允许超出根定义范围的操作。
5. **模板支持**：为常见引导场景提供结构化的模板。

## 练习

1. **文件范围限定器**：创建一个使用根将文件操作限制在特定目录的服务器。

2. **智能引导器**：实现一个在用户提供模糊请求时提出澄清性问题的工具。

3. **多根管理器**：构建一个支持和管理多个根的服务器，每个根用于不同的服务。

4. **渐进式引导流程**：创建一系列引导步骤，逐步完善复杂的用户请求。

5. **根感知安全**：实现一个使用根来强制执行访问控制和审计日志的安全层。

## 术语表

- **根**：定义服务器操作范围的边界。
- **引导**：通过提问完善用户请求的过程。
- **范围**：服务器允许操作的定义边界。
- **边界**：根定义的允许访问的逻辑边界。
- **模板**：用于引导常见场景的结构化格式。

## 延伸阅读

- MCP 根规范
- MCP 客户端-服务器初始化
- 安全边界设计模式
- 提示词工程最佳实践
