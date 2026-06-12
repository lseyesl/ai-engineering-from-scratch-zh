# 技能与代理 SDK

> 技能和代理 SDK 为构建、共享和组合 AI 代理能力提供了标准化的框架，使代理开发模块化且可复用。

**类型：** 构建（Build）
**语言：** TypeScript
**前置要求：** MCP 基础，A2A 协议知识
**时间：** ~50 分钟

## 学习目标

- 理解技能系统架构及其与 MCP/A2A 的关系
- 使用标准接口设计和实现可复用技能
- 构建用于代理创建的 SDK
- 组合多个技能以构建复杂代理
- 发布和发现共享技能

## 什么是技能？

技能是模块化的、可复用的能力单元，可以插入到 AI 代理中。它们类似于 ChatGPT 的插件或移动应用——每个技能为代理添加特定的功能。

```
┌─────────────────────────────────────────┐
│             AI 代理                       │
│                                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ 技能 A   │ │ 技能 B   │ │ 技能 C   │   │
│  │ (搜索)   │ │ (代码)   │ │ (文件)   │   │
│  └──────────┘ └──────────┘ └──────────┘   │
│       │            │            │          │
│       ▼            ▼            ▼          │
│  ┌──────────────────────────────────────┐  │
│  │          MCP 传输层                   │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### 技能与 MCP 工具

| 方面 | MCP 工具 | 技能 |
|------|---------|-------|
| 粒度 | 原子操作 | 组合能力 |
| 状态 | 无状态 | 有状态可能 |
| 配置 | 每个请求的参数 | 声明式配置 |
| 组合 | 通过客户端编排 | 内建组合 |
| 发现 | 服务器能力列举 | 技能注册中心 |

## 技能接口

```typescript
// skill-types.ts
// 技能配置
interface SkillConfig {
  name: string;
  version: string;
  description: string;
  author?: string;
  dependencies?: string[];         // 依赖的技能
  settings?: Record<string, SkillSetting>;
}

interface SkillSetting {
  type: "string" | "number" | "boolean" | "select";
  label: string;
  description: string;
  default?: unknown;
  required?: boolean;
  options?: string[];              // select 类型
  secret?: boolean;                // 是否敏感
}

// 技能上下文
interface SkillContext {
  agentId: string;
  config: Record<string, unknown>;
  logger: {
    info: (msg: string) => void;
    error: (msg: string) => void;
    warn: (msg: string) => void;
  };
  storage: SkillStorage;
  events: SkillEventBus;
}

// 技能存储
interface SkillStorage {
  get(key: string): Promise<unknown>;
  set(key: string, value: unknown): Promise<void>;
  delete(key: string): Promise<void>;
  list(prefix: string): Promise<string[]>;
}

// 技能事件
interface SkillEventBus {
  emit(event: string, data: unknown): void;
  on(event: string, handler: (data: unknown) => void): void;
}

// 基本技能类
abstract class Skill {
  public config: SkillConfig;
  protected context: SkillContext | null = null;

  constructor(config: SkillConfig) {
    this.config = config;
  }

  // 初始化技能
  abstract initialize(context: SkillContext): Promise<void>;

  // 执行技能
  abstract execute(
    input: unknown,
    options?: ExecuteOptions
  ): Promise<SkillResult>;

  // 获取技能能力
  abstract getCapabilities(): SkillCapability[];

  // 清理
  abstract cleanup(): Promise<void>;

  // 验证配置
  validateConfig(config: Record<string, unknown>): boolean {
    for (const [key, setting] of Object.entries(this.config.settings || {})) {
      if (setting.required && config[key] === undefined) {
        return false;
      }
    }
    return true;
  }
}

interface SkillCapability {
  name: string;
  description: string;
  inputSchema: object;
  outputSchema: object;
}

interface ExecuteOptions {
  timeout?: number;
  priority?: "low" | "normal" | "high";
  signal?: AbortSignal;
}

interface SkillResult {
  success: boolean;
  data: unknown;
  error?: string;
  metrics?: {
    duration: number;
    tokensUsed?: number;
  };
}
```

## 实现一个技能

```typescript
// search-skill.ts
class SearchSkill extends Skill {
  private apiKey: string = "";
  private maxResults: number = 5;

  constructor() {
    super({
      name: "search",
      version: "1.0.0",
      description: "搜索网络并返回结果",
      settings: {
        apiKey: {
          type: "string",
          label: "搜索 API 密钥",
          description: "搜索服务 API 密钥",
          required: true,
          secret: true,
        },
        maxResults: {
          type: "number",
          label: "最大结果数",
          description: "每次搜索返回的最大结果数",
          default: 5,
          required: false,
        },
        searchProvider: {
          type: "select",
          label: "搜索提供商",
          description: "使用的搜索服务",
          default: "web",
          options: ["web", "news", "academic"],
        },
      },
    });
  }

  async initialize(context: SkillContext): Promise<void> {
    this.context = context;

    // 从配置加载设置
    this.apiKey = String(context.config["apiKey"] || "");
    this.maxResults = Number(context.config["maxResults"] || 5);

    if (!this.apiKey) {
      throw new Error("SearchSkill：需要 API 密钥");
    }

    context.logger.info("SearchSkill 初始化完成");
  }

  getCapabilities(): SkillCapability[] {
    return [
      {
        name: "web_search",
        description: "搜索网络上的信息",
        inputSchema: {
          type: "object",
          properties: {
            query: { type: "string", description: "搜索查询" },
            count: { type: "number", description: "结果数量" },
          },
          required: ["query"],
        },
        outputSchema: {
          type: "array",
          items: {
            type: "object",
            properties: {
              title: { type: "string" },
              url: { type: "string" },
              snippet: { type: "string" },
            },
          },
        },
      },
    ];
  }

  async execute(
    input: SearchInput,
    options?: ExecuteOptions
  ): Promise<SkillResult> {
    const startTime = Date.now();

    try {
      const query = typeof input === "string" ? input : input.query;
      const count = (input as SearchInput).count || this.maxResults;

      // 执行搜索
      const results = await this.performSearch(query, count);

      return {
        success: true,
        data: results,
        metrics: {
          duration: Date.now() - startTime,
        },
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: String(error),
        metrics: { duration: Date.now() - startTime },
      };
    }
  }

  private async performSearch(
    query: string,
    count: number
  ): Promise<SearchResult[]> {
    // 模拟搜索结果
    return [
      {
        title: `${query} 的结果 1`,
        url: `https://example.com/result-1`,
        snippet: `这是 ${query} 的搜索结果摘要。`,
      },
      {
        title: `${query} 的结果 2`,
        url: `https://example.com/result-2`,
        snippet: `另一个关于 ${query} 的结果。`,
      },
    ];
  }

  async cleanup(): Promise<void> {
    this.context?.logger.info("SearchSkill 已清理");
  }
}

interface SearchInput {
  query: string;
  count?: number;
}

interface SearchResult {
  title: string;
  url: string;
  snippet: string;
}
```

## 代理 SDK

```typescript
// agent-sdk.ts
type SkillConstructor = new () => Skill;

class AgentSDK {
  // 注册的技能工厂
  private static skillRegistry: Map<string, SkillConstructor> = new Map();

  // 注册一个技能
  static registerSkill(name: string, ctor: SkillConstructor) {
    this.skillRegistry.set(name, ctor);
    console.log(`技能已注册：${name}`);
  }

  // 创建技能实例
  static createSkill(name: string, config: SkillConfig): Skill {
    const ctor = this.skillRegistry.get(name);
    if (!ctor) {
      throw new Error(`未注册的技能：${name}`);
    }
    return new ctor();
  }
}

// 代理构建器
class AgentBuilder {
  private name: string = "";
  private skills: Skill[] = [];
  private configs: Map<string, Record<string, unknown>> = new Map();
  private storage: SkillStorage = new MemoryStorage();
  private eventBus: SkillEventBus = new SimpleEventBus();

  constructor(name: string) {
    this.name = name;
  }

  // 添加技能
  withSkill(skillName: string, config?: Record<string, unknown>): AgentBuilder {
    const skill = AgentSDK.createSkill(skillName, {
      name: skillName,
      version: "1.0.0",
      description: "",
    });
    this.skills.push(skill);
    if (config) {
      this.configs.set(skillName, config);
    }
    return this;
  }

  // 添加自定义技能实例
  withCustomSkill(skill: Skill, config?: Record<string, unknown>): AgentBuilder {
    this.skills.push(skill);
    if (config) {
      this.configs.set(skill.config.name, config);
    }
    return this;
  }

  // 构建代理
  async build(): Promise<Agent> {
    const agent = new Agent(this.name);

    for (const skill of this.skills) {
      const context: SkillContext = {
        agentId: agent.id,
        config: this.configs.get(skill.config.name) || {},
        logger: console,
        storage: this.storage,
        events: this.eventBus,
      };
      await skill.initialize(context);
      agent.registerSkill(skill);
    }

    return agent;
  }
}

// 代理类
class Agent {
  public id: string;
  public name: string;
  private skills: Map<string, Skill> = new Map();
  private eventBus: SkillEventBus;

  constructor(name: string) {
    this.id = `agent-${crypto.randomUUID().slice(0, 8)}`;
    this.name = name;
    this.eventBus = new SimpleEventBus();
  }

  registerSkill(skill: Skill) {
    this.skills.set(skill.config.name, skill);
  }

  // 执行技能
  async executeSkill(
    skillName: string,
    input: unknown
  ): Promise<SkillResult> {
    const skill = this.skills.get(skillName);
    if (!skill) {
      throw new Error(`技能未找到：${skillName}`);
    }

    this.eventBus.emit("skill:before", { skill: skillName, input });
    const result = await skill.execute(input);
    this.eventBus.emit("skill:after", { skill: skillName, result });

    return result;
  }

  // 列出可用技能
  listSkills(): SkillConfig[] {
    return Array.from(this.skills.values()).map((s) => s.config);
  }

  // 获取能力
  getCapabilities(): SkillCapability[] {
    return Array.from(this.skills.values()).flatMap((s) =>
      s.getCapabilities()
    );
  }

  // 清理所有技能
  async cleanup(): Promise<void> {
    for (const skill of this.skills.values()) {
      await skill.cleanup();
    }
  }
}

// 内存存储实现
class MemoryStorage implements SkillStorage {
  private store: Map<string, unknown> = new Map();

  async get(key: string): Promise<unknown> {
    return this.store.get(key);
  }
  async set(key: string, value: unknown): Promise<void> {
    this.store.set(key, value);
  }
  async delete(key: string): Promise<void> {
    this.store.delete(key);
  }
  async list(prefix: string): Promise<string[]> {
    return Array.from(this.store.keys()).filter((k) =>
      k.startsWith(prefix)
    );
  }
}

// 简单事件总线
class SimpleEventBus implements SkillEventBus {
  private handlers: Map<string, Array<(data: unknown) => void>> = new Map();

  emit(event: string, data: unknown): void {
    const handlers = this.handlers.get(event);
    if (handlers) {
      for (const handler of handlers) {
        handler(data);
      }
    }
  }

  on(event: string, handler: (data: unknown) => void): void {
    const handlers = this.handlers.get(event) || [];
    handlers.push(handler);
    this.handlers.set(event, handlers);
  }
}
```

## 多技能代理示例

```typescript
// multi-skill-agent.ts
// 注册技能
AgentSDK.registerSkill("search", SearchSkill);

// 创建带有多个技能的代理
async function createMultiSkillAgent() {
  const agent = await new AgentBuilder("研究助手")
    .withSkill("search", {
      apiKey: "sk-xxx",
      maxResults: 10,
    })
    .withSkill("code", {
      language: "typescript",
      formatter: "prettier",
    })
    .withSkill("files", {
      allowedPaths: ["/home/user/projects"],
    })
    .build();

  console.log(`代理 ${agent.name} (${agent.id}) 已创建`);
  console.log("可用技能：", agent.listSkills().map((s) => s.name));

  // 使用代理
  const result = await agent.executeSkill("search", {
    query: "最新的 TypeScript 特性 2026",
    count: 5,
  });

  console.log("搜索结果：", result.data);

  return agent;
}
```

## 技能注册中心（发布/发现）

```typescript
// skill-registry.ts
interface SkillManifest {
  name: string;
  version: string;
  description: string;
  author: string;
  capabilities: string[];
  settings: SkillSetting[];
  repository?: string;
  readme?: string;
}

class SkillRegistry {
  private skills: Map<string, SkillManifest> = new Map();

  // 发布技能
  publish(manifest: SkillManifest) {
    const key = `${manifest.name}@${manifest.version}`;
    this.skills.set(key, manifest);
    console.log(`技能已发布：${key}`);
  }

  // 发现技能
  search(query: string): SkillManifest[] {
    return Array.from(this.skills.values()).filter(
      (s) =>
        s.name.includes(query) ||
        s.description.includes(query) ||
        s.capabilities.some((c) => c.includes(query))
    );
  }

  // 获取技能详情
  get(name: string, version?: string): SkillManifest | null {
    // 获取最新版本或指定版本
    const allVersions = Array.from(this.skills.keys())
      .filter((k) => k.startsWith(name + "@"))
      .sort()
      .reverse();

    const targetVersion = version || allVersions[0];
    return targetVersion ? this.skills.get(targetVersion) || null : null;
  }

  // 列出所有技能
  list(): SkillManifest[] {
    // 只返回每个技能的最新版本
    const latest = new Map<string, SkillManifest>();
    for (const [key, manifest] of this.skills) {
      const name = key.split("@")[0];
      const existing = latest.get(name);
      if (!existing || manifest.version > existing.version) {
        latest.set(name, manifest);
      }
    }
    return Array.from(latest.values());
  }
}
```

## 与 MCP 集成

```typescript
// mcp-skill-bridge.ts
// 将 MCP 工具暴露为技能
class MCPToolSkill extends Skill {
  private mcpClient: any;

  constructor(toolName: string, client: any) {
    super({
      name: `mcp-${toolName}`,
      version: "1.0.0",
      description: `MCP 工具：${toolName}`,
    });
    this.mcpClient = client;
  }

  async initialize(context: SkillContext): Promise<void> {
    this.context = context;
  }

  getCapabilities(): SkillCapability[] {
    return [
      {
        name: this.config.name,
        description: this.config.description,
        inputSchema: { type: "object" },
        outputSchema: { type: "object" },
      },
    ];
  }

  async execute(input: unknown): Promise<SkillResult> {
    const startTime = Date.now();
    try {
      const result = await this.mcpClient.callTool(
        this.config.name.replace("mcp-", ""),
        input
      );
      return {
        success: true,
        data: result,
        metrics: { duration: Date.now() - startTime },
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: String(error),
        metrics: { duration: Date.now() - startTime },
      };
    }
  }

  async cleanup(): Promise<void> {}
}
```

## 最佳实践

1. **技能原子性**：每个技能应做好一件事。避免创建做太多事情的整体技能。
2. **声明式配置**：技能通过声明式配置初始化，而非硬编码值。
3. **优雅降级**：技能应在依赖失败时优雅降级。
4. **事件驱动通信**：技能使用事件总线通信，不应直接互相引用。
5. **版本兼容性**：语义化版本控制和版本验证以防止破坏性变更。

## 练习

1. **天气技能**：实现一个获取天气数据的技能（真实的或被模拟的）。

2. **代理组合**：创建一个整合搜索、摘要和格式化技能的代理。

3. **技能注册中心 Web UI**：构建一个用于浏览、安装和配置技能的 Web 界面。

4. **MCP 技能桥接**：创建一个自动将 MCP 工具转换为技能的技能适配器。

5. **技能依赖解析**：实现当技能依赖于其他技能时的依赖解析系统。

## 术语表

- **技能**：可以插入代理的模块化、可复用能力单元。
- **代理 SDK**：用于构建和组织代理技能的框架。
- **技能注册中心**：可用技能及其版本的中央目录。
- **能力**：技能可以执行的特定功能。
- **声明式配置**：描述技能应如何表现的配置方法。

## 延伸阅读

- OpenAI 插件系统
- 模块化 AI 架构
- 微内核架构模式
- 依赖注入模式
