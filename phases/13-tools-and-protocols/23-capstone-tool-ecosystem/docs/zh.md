# 顶点项目：工具生态系统

> 结合 MCP、A2A、路由、安全和可观测性来构建一个完整的、生产级的 AI 工具生态系统。

**类型：** 构建（Build）
**语言：** TypeScript
**前置要求：** 完成 Phase 13 的所有课程
**时间：** ~90 分钟

## 学习目标

- 集成 MCP、A2A 和路由层为统一系统
- 实现生产级安全、监控和速率限制
- 构建一个具有多种工具和专业代理的工具生态系统
- 创建全面的监控和可观测性
- 部署和操作工具生态系统

## 项目概述

在这个顶点项目中，你将构建"Galaxy"——一个完整的 AI 工具生态系统，结合了你在 Phase 13 中学到的所有概念：

```
┌──────────────────────────────────────────────────┐
│                Galaxy 工具生态系统                   │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ 客户端   │  │ 客户端   │  │ 客户端   │         │
│  │ (CLI)    │  │ (Web)    │  │ (API)    │         │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘         │
│        │              │              │              │
│        ▼              ▼              ▼              │
│  ┌──────────────────────────────────────────┐      │
│  │           API 网关                        │      │
│  │  (认证、速率限制、路由、审计)              │      │
│  └──────────┬──────────────────┬────────────┘      │
│             │                  │                    │
│  ┌──────────▼────┐    ┌───────▼──────────┐        │
│  │  MCP 服务器们  │    │  A2A 代理网络     │        │
│  │               │    │                  │        │
│  │  • 文件系统   │    │  • 研究代理      │        │
│  │  • 数据库     │    │  • 编码代理      │        │
│  │  • 搜索       │    │  • 审查代理      │        │
│  │  • 计算器     │    │  • 测试代理      │        │
│  └───────────────┘    └──────────────────┘        │
│                                                    │
│  ┌──────────────────────────────────────────┐      │
│  │     可观测性栈                              │      │
│  │  (OpenTelemetry、追踪、指标、日志)          │      │
│  └──────────────────────────────────────────┘      │
└────────────────────────────────────────────────────┘
```

## 第 1 步：MCP 工具服务器

首先创建作为生态系统基础的 MCP 服务器：

```typescript
// tools/file-system-server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { readFile, writeFile, readdir } from "fs/promises";
import { resolve, relative } from "path";

export class FileSystemServer {
  private server: Server;
  private allowedBaseDir: string;

  constructor(baseDir: string) {
    this.allowedBaseDir = resolve(baseDir);
    this.server = new Server(
      { name: "fs-server", version: "1.0.0" },
      { capabilities: { tools: {} } }
    );
    this.setupHandlers();
  }

  private setupHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: "read_file",
          description: "读取文件内容",
          inputSchema: {
            type: "object",
            properties: {
              path: { type: "string", description: "文件路径" },
            },
            required: ["path"],
          },
        },
        {
          name: "write_file",
          description: "写入文件",
          inputSchema: {
            type: "object",
            properties: {
              path: { type: "string", description: "文件路径" },
              content: { type: "string", description: "文件内容" },
            },
            required: ["path", "content"],
          },
        },
        {
          name: "list_dir",
          description: "列出目录内容",
          inputSchema: {
            type: "object",
            properties: {
              path: { type: "string", description: "目录路径" },
            },
            required: ["path"],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;
      const safePath = this.resolveSafePath(String(args?.path || ""));

      switch (name) {
        case "read_file": {
          const content = await readFile(safePath, "utf-8");
          return { content: [{ type: "text", text: content }] };
        }
        case "write_file": {
          await writeFile(safePath, String(args?.content || ""));
          return { content: [{ type: "text", text: "文件已写入" }] };
        }
        case "list_dir": {
          const entries = await readdir(safePath);
          return {
            content: [{ type: "text", text: entries.join("\n") }],
          };
        }
        default:
          throw new Error(`未知工具：${name}`);
      }
    });
  }

  private resolveSafePath(requestedPath: string): string {
    const resolved = resolve(this.allowedBaseDir, requestedPath);
    if (!resolved.startsWith(this.allowedBaseDir)) {
      throw new Error(`路径超出范围：${requestedPath}`);
    }
    return resolved;
  }

  async connect(transport: StdioServerTransport) {
    await this.server.connect(transport);
  }
}
```

## 第 2 步：A2A 代理网络

接下来，创建可以相互协作的专业 AI 代理：

```typescript
// agents/research-agent.ts
import { A2AAgent } from "../a2a/a2a-agent.js";

class ResearchAgent extends A2AAgent {
  private searchResults: Map<string, unknown[]> = new Map();

  constructor(registryEndpoint: string) {
    super({
      agentId: "researcher-1",
      name: "Research Agent",
      capabilities: [
        {
          name: "web_research",
          version: "1.0",
          description: "对主题进行网络研究并总结发现",
          inputSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
          outputSchema: { type: "object", properties: { summary: { type: "string" }, sources: { type: "array" } } },
        },
        {
          name: "fact_check",
          version: "1.0",
          description: "根据可靠来源核实声明",
          inputSchema: { type: "object", properties: { claim: { type: "string" } }, required: ["claim"] },
          outputSchema: { type: "object", properties: { verified: { type: "boolean" }, evidence: { type: "string" } } },
        },
      ],
      registryEndpoint,
    });
  }

  protected async processTask(capability: string, parameters: unknown): Promise<unknown> {
    switch (capability) {
      case "web_research": {
        const { topic } = parameters as { topic: string };
        return this.performResearch(topic);
      }
      case "fact_check": {
        const { claim } = parameters as { claim: string };
        return this.performFactCheck(claim);
      }
      default:
        throw new Error(`不支持的 capability：${capability}`);
    }
  }

  private async performResearch(topic: string) {
    // 模拟研究
    await new Promise(r => setTimeout(r, 3000));
    return {
      summary: `关于"${topic}"的研究发现：
- 发现 3 个主要来源
- 识别出 2 个关键主题
- 找到 5 个相关引用`,
      sources: [
        { title: `${topic} 概述`, url: "https://example.com/overview", relevance: 0.95 },
        { title: `${topic} 最新发展`, url: "https://example.com/latest", relevance: 0.88 },
      ],
      timestamp: new Date().toISOString(),
    };
  }

  private async performFactCheck(claim: string) {
    // 模拟事实核查
    return {
      claim,
      verified: true,
      confidence: 0.92,
      evidence: `在 3 个独立来源中找到支持该声明的证据。`,
      sources: ["https://source1.com", "https://source2.com"],
    };
  }
}
```

```typescript
// agents/coding-agent.ts
class CodingAgent extends A2AAgent {
  constructor(registryEndpoint: string) {
    super({
      agentId: "coder-1",
      name: "Coding Agent",
      capabilities: [
        {
          name: "code_review",
          version: "1.0",
          description: "审查代码并建议改进",
          inputSchema: { type: "object", properties: { code: { type: "string" }, language: { type: "string" } }, required: ["code"] },
          outputSchema: { type: "object" },
        },
        {
          name: "code_generation",
          version: "1.0",
          description: "从描述生成代码",
          inputSchema: { type: "object", properties: { description: { type: "string" }, language: { type: "string" } }, required: ["description"] },
          outputSchema: { type: "object" },
        },
      ],
      registryEndpoint,
    });
  }

  protected async processTask(capability: string, parameters: unknown): Promise<unknown> {
    if (capability === "code_review") {
      const { code, language } = parameters as { code: string; language: string };
      return {
        review: `审查结果（${language}）：
- 代码质量：良好
- 潜在问题：在错误处理中发现了 1 个边缘情况
- 建议：考虑添加输入验证`,
        issues: [{ severity: "warning", line: 15, message: "缺少 null 检查" }],
      };
    }
    throw new Error(`不支持的 capability：${capability}`);
  }
}
```

## 第 3 步：LLM 路由器

为生态系统添加智能 LLM 路由：

```typescript
// routing/ecosystem-router.ts
import { LLMRouter } from "../routing/llm-router.js";
import { CostTracker } from "../routing/cost-tracker.js";

class EcosystemRouter {
  private router: LLMRouter;
  private costTracker: CostTracker;

  constructor() {
    this.router = new LLMRouter();
    this.costTracker = new CostTracker(500); // 500 美元月度预算
  }

  async routeAgentRequest(
    agentId: string,
    taskType: string,
    complexity: "simple" | "medium" | "complex"
  ) {
    const startTime = Date.now();

    const routerRequest = {
      messages: [
        { role: "user", content: `在 ${agentId} 上执行 ${taskType} 任务` },
      ],
      maxTokens: complexity === "complex" ? 4000 : 1000,
      priority: (complexity === "complex" ? "high" : "medium") as
        | "low"
        | "medium"
        | "high",
    };

    try {
      const response = await this.router.route(routerRequest);

      this.costTracker.trackCall({
        model: response.model,
        promptTokens: response.usage.promptTokens,
        completionTokens: response.usage.completionTokens,
        cost: response.usage.cost,
        userId: agentId,
        taskType,
      });

      return {
        ...response,
        routingLatency: Date.now() - startTime,
      };
    } catch (error) {
      // 回退到本地处理
      return {
        model: "local-fallback",
        provider: "local",
        content: "处理中，使用本地回退...",
        usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0, cost: 0 },
        latency: Date.now() - startTime,
        attemptedModels: [],
      };
    }
  }
}
```

## 第 4 步：网关与安全

实现 API 网关和安全层：

```typescript
// gateway/ecosystem-gateway.ts
import express from "express";
import { MCPGateway } from "../gateway/mcp-gateway.js";
import { AuditLogger } from "../auth/audit-log.js";
import { MCPSecurityMiddleware } from "../security/security-middleware.js";

class EcosystemGateway {
  private gateway: MCPGateway;
  private audit: AuditLogger;
  private security: MCPSecurityMiddleware;
  private router: EcosystemRouter;
  private app: express.Application;

  constructor() {
    this.gateway = new MCPGateway();
    this.audit = new AuditLogger("/var/log/galaxy/audit.log");
    this.security = new MCPSecurityMiddleware({
      maxInputSize: 1_000_000,
      rateLimitPerMinute: 100,
      allowedClients: ["*"],
      blockedOperations: [],
    });
    this.router = new EcosystemRouter();
    this.app = express();
    this.setupRoutes();
  }

  private setupRoutes() {
    this.app.use(express.json());

    // 健康检查
    this.app.get("/health", (req, res) => {
      res.json({
        status: "healthy",
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
      });
    });

    // 工具执行端点
    this.app.post("/tools/:toolName", async (req, res) => {
      const startTime = Date.now();

      try {
        const { toolName } = req.params;
        const clientId = req.headers["x-client-id"] as string || "anonymous";

        // 安全检查
        const validation = this.security.validateRequest({
          method: "tools/call",
          params: { name: toolName, arguments: req.body },
        });

        if (!validation.valid) {
          this.audit.log({
            clientId,
            userId: clientId,
            action: "tools/call",
            resource: toolName,
            result: "denied",
            details: { reason: validation.error },
            ip: req.ip,
            userAgent: req.headers["user-agent"] || "",
          });
          return res.status(403).json({ error: validation.error });
        }

        // 速率限制
        if (!this.security.checkRateLimit(clientId)) {
          return res.status(429).json({ error: "速率限制超出" });
        }

        // 使用 LLM 路由器确定最佳处理路径
        const routing = await this.router.routeAgentRequest(
          clientId,
          toolName,
          "medium"
        );

        // 执行工具
        const result = await this.gateway.executeToolSafely(
          toolName,
          req.body,
          { clientId }
        );

        // 审计
        this.audit.log({
          clientId,
          userId: clientId,
          action: "tools/call",
          resource: toolName,
          result: "success",
          details: { duration: Date.now() - startTime, routing },
          ip: req.ip,
          userAgent: req.headers["user-agent"] || "",
        });

        res.json({
          result,
          meta: {
            duration: Date.now() - startTime,
            model: routing.model,
          },
        });
      } catch (error) {
        res.status(500).json({
          error: "内部错误",
          message: String(error),
        });
      }
    });

    // A2A 代理通信
    this.app.post("/agents/:agentId/message", async (req, res) => {
      // 将消息转发到 A2A 代理
      res.json({ received: true, agentId: req.params.agentId });
    });

    // 指标端点
    this.app.get("/metrics", async (req, res) => {
      res.json({
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        requests: this.audit.search({}).length,
      });
    });
  }

  async start(port: number) {
    return new Promise((resolve) => {
      this.app.listen(port, () => {
        console.log(`Galaxy 生态系统网关运行在端口 ${port}`);
        resolve(true);
      });
    });
  }
}
```

## 第 5 步：可观测性设置

```typescript
// observability/telemetry.ts
import { trace, metrics } from "@opentelemetry/api";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { MeterProvider } from "@opentelemetry/sdk-metrics";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-http";
import { PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import { Resource } from "@opentelemetry/resources";

class GalaxyTelemetry {
  private tracerProvider: NodeTracerProvider;
  private meterProvider: MeterProvider;

  constructor() {
    const resource = new Resource({
      "service.name": "galaxy-ecosystem",
      "service.version": "1.0.0",
      "deployment.environment": process.env.NODE_ENV || "development",
    });

    // 设置追踪
    this.tracerProvider = new NodeTracerProvider({ resource });

    if (process.env.OTEL_EXPORTER_OTLP_ENDPOINT) {
      this.tracerProvider.addSpanProcessor(
        new SimpleSpanProcessor(
          new OTLPTraceExporter({
            url: `${process.env.OTEL_EXPORTER_OTLP_ENDPOINT}/v1/traces`,
          })
        )
      );
    }

    this.tracerProvider.register();

    // 设置指标
    this.meterProvider = new MeterProvider({ resource });

    if (process.env.OTEL_EXPORTER_OTLP_ENDPOINT) {
      this.meterProvider.addMetricReader(
        new PeriodicExportingMetricReader({
          exporter: new OTLPMetricExporter({
            url: `${process.env.OTEL_EXPORTER_OTLP_ENDPOINT}/v1/metrics`,
          }),
          exportIntervalMillis: 60_000,
        })
      );
    }

    metrics.setGlobalMeterProvider(this.meterProvider);
  }

  getTracer() {
    return trace.getTracer("galaxy");
  }

  getMeter() {
    return metrics.getMeter("galaxy");
  }
}
```

## 第 6 步：主启动脚本

```typescript
// index.ts - Galaxy 生态系统入口点
import { FileSystemServer } from "./tools/file-system-server.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { EcosystemGateway } from "./gateway/ecosystem-gateway.js";
import { GalaxyTelemetry } from "./observability/telemetry.js";
import { SkillRegistry } from "./skills/skill-registry.js";
import { CostTracker } from "./routing/cost-tracker.js";

class GalaxyEcosystem {
  private gateway: EcosystemGateway;
  private telemetry: GalaxyTelemetry;
  private registry: SkillRegistry;
  private costTracker: CostTracker;
  private fsServer: FileSystemServer;
  private agents: A2AAgent[] = [];

  constructor() {
    console.log("正在启动 Galaxy 工具生态系统...\n");

    // 初始化组件
    this.telemetry = new GalaxyTelemetry();
    this.registry = new SkillRegistry();
    this.costTracker = new CostTracker(1000);
    this.fsServer = new FileSystemServer("/home/user/projects");
    this.gateway = new EcosystemGateway();
  }

  async initialize() {
    // 1. 启动 MCP 服务器
    console.log("正在启动 MCP 文件系统服务器...");
    const transport = new StdioServerTransport();
    await this.fsServer.connect(transport);

    // 2. 注册 A2A 代理
    console.log("正在注册 A2A 代理...");
    const researchAgent = new ResearchAgent("http://localhost:4000");
    const codingAgent = new CodingAgent("http://localhost:4000");
    await researchAgent.announce();
    await codingAgent.announce();
    this.agents.push(researchAgent, codingAgent);

    // 3. 启动网关
    console.log("正在启动 API 网关...");
    await this.gateway.start(3000);

    // 4. 注册内置技能
    console.log("正在注册技能...");
    this.registry.publish({
      name: "web_search",
      version: "1.0.0",
      description: "搜索网络",
      author: "galaxy-team",
      capabilities: ["search", "research"],
      settings: [
        { type: "string", label: "API Key", description: "搜索 API 密钥", required: true, secret: true },
        { type: "number", label: "Max Results", description: "最大结果数", default: 5 },
      ],
    });

    // 5. 显示仪表盘信息
    this.printDashboard();
  }

  private printDashboard() {
    console.log("\n┌────────────────────────────────────────────┐");
    console.log("│        Galaxy 工具生态系统                 │");
    console.log("│                                            │");
    console.log("│  📡 网关:       http://localhost:3000       │");
    console.log("│  🔗 健康检查:   http://localhost:3000/health │");
    console.log("│  📊 指标:       http://localhost:3000/metrics │");
    console.log("│  🤖 代理:       " + this.agents.length + " 在线                 │");
    console.log("│  🛠️  技能:       " + this.registry.list().length + " 已注册               │");
    console.log("│  💰 预算:       $" + this.costTracker.getCostByModel() + " / $1000          │");
    console.log("└────────────────────────────────────────────┘\n");
  }

  async shutdown() {
    console.log("正在关闭 Galaxy 生态系统...");
    for (const agent of this.agents) {
      await agent.shutdown();
    }
    process.exit(0);
  }
}

// 运行生态系统
const ecosystem = new GalaxyEcosystem();
ecosystem.initialize().catch(console.error);

// 优雅关闭
process.on("SIGINT", () => ecosystem.shutdown());
process.on("SIGTERM", () => ecosystem.shutdown());
```

## 运行生态系统

```bash
# 安装依赖
npm install @modelcontextprotocol/sdk @opentelemetry/api @opentelemetry/sdk-trace-node

# 启动生态系统
npm run start

# 测试健康检查
curl http://localhost:3000/health

# 测试工具执行
curl -X POST http://localhost:3000/tools/read_file \
  -H "Content-Type: application/json" \
  -H "x-client-id: test-client" \
  -d '{"path": "./README.md"}'
```

## 练习

1. **添加新工具**：向生态系统添加数据库 MCP 服务器（SQLite 查询）。

2. **代理协作**：实现一个跨越研究代理和编码代理的多步骤工作流（研究→设计→实现）。

3. **自定义仪表盘**：构建一个使用网关指标端点的实时生态系统监控仪表盘。

4. **技能市场**：创建一个可供用户在运行时浏览和安装技能的 Web UI。

5. **负载测试**：编写一个对生态系统进行负载测试并衡量其扩展能力的脚本。

## 项目扩展思路

- **插件系统**：允许第三方开发者创建和发布工具
- **代理市场**：用户可以浏览、评级和安装专业代理
- **工作流编辑器**：用于创建多步骤代理工作流的可视化编辑器
- **成本分析**：详细的成本分解和使用预测
- **A/B 测试框架**：并排比较不同模型和工具配置

## 术语表

- **Galaxy**：本顶点项目的代号，代表一个完整的 AI 工具生态系统。
- **生态系统**：一组集成的工具、代理和服务协同工作。
- **网关**：生态系统的中央入口点，处理所有传入的请求。
- **代理网络**：可以协作完成任务的 AI 代理集合。
- **可观测性栈**：用于监控、追踪和调试生态系统的工具集合。

## 延伸阅读

- MCP 规范
- A2A 协议设计
- OpenTelemetry 文档
- 微服务架构模式
- 12 Factor 应用方法论
