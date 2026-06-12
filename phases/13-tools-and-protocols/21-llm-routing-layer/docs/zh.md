# LLM 路由层

> LLM 路由层智能地将请求分发到最合适的模型和端点，优化成本、延迟和响应质量。

**类型：** 构建（Build）
**语言：** TypeScript
**前置要求：** 生成式 AI 基础，MCP 概念
**时间：** ~50 分钟

## 学习目标

- 理解 LLM 路由的需求和策略
- 实现基于规则和基于质量的路由器
- 构建回退链以提高可靠性
- 实现成本感知的路由决策
- 将 LLM 路由与 MCP 工具执行集成

## 为什么需要 LLM 路由？

不同的 LLM 在不同任务上表现出色。路由器确保每个请求被发送到最合适的模型：

```
请求进入
    │
    ▼
┌───────────────────┐
│  LLM 路由器        │
│                   │
│  ┌─────────────┐  │  ┌──────────┐
│  │ 规则引擎     │───>│ GPT-4o   │ ← 高智能任务
│  ├─────────────┤  │  ├──────────┤
│  │ 成本分析     │───>│ Claude   │ ← 长上下文任务
│  ├─────────────┤  │  ├──────────┤
│  │ 延迟检查     │───>│ GPT-4o-  │ ← 快速/简单任务
│  └─────────────┘  │  │ Mini     │
│                   │  └──────────┘
└───────────────────┘
```

### 路由策略

| 策略 | 描述 | 使用场景 |
|----------|-------------|-----------|
| 基于规则 | 匹配请求特征到模型 | 已知路由决策 |
| 基于质量 | 使用 LLM 判断哪个模型最适合 | 复杂路由决策 |
| 成本感知 | 在质量和成本间取得平衡 | 预算受限场景 |
| 延迟感知 | 最小化响应时间 | 用户面向应用 |
| 回退链 | 尝试备选模型 | 高可靠性需求 |

## 路由器配置

```typescript
// router-config.ts
interface ModelConfig {
  name: string;
  provider: "openai" | "anthropic" | "ollama" | "custom";
  model: string;
  capabilities: {
    maxTokens: number;
    supportsTools: boolean;
    supportsVision: boolean;
    supportsStreaming: boolean;
  };
  cost: {
    perPromptToken: number;     // 美元
    perCompletionToken: number; // 美元
  };
  performance: {
    avgLatencyMs: number;
    p95LatencyMs: number;
  };
}

interface RoutingRule {
  name: string;
  priority: number;          // 优先级（越高越优先）
  condition: RoutingCondition;
  targetModel: string;       // 引用模型名称
}

type RoutingCondition =
  | { type: "task_type"; values: string[] }
  | { type: "max_tokens"; max: number }
  | { type: "requires_tools"; value: boolean }
  | { type: "requires_vision"; value: boolean }
  | { type: "max_cost"; maxUsd: number }
  | { type: "max_latency"; maxMs: number }
  | { type: "model_preference"; models: string[] };

const MODELS: ModelConfig[] = [
  {
    name: "gpt-4o",
    provider: "openai",
    model: "gpt-4o-2024-05-13",
    capabilities: {
      maxTokens: 128000,
      supportsTools: true,
      supportsVision: true,
      supportsStreaming: true,
    },
    cost: {
      perPromptToken: 0.000005,
      perCompletionToken: 0.000015,
    },
    performance: {
      avgLatencyMs: 2000,
      p95LatencyMs: 5000,
    },
  },
  {
    name: "gpt-4o-mini",
    provider: "openai",
    model: "gpt-4o-mini-2024-07-18",
    capabilities: {
      maxTokens: 128000,
      supportsTools: true,
      supportsVision: true,
      supportsStreaming: true,
    },
    cost: {
      perPromptToken: 0.00000015,
      perCompletionToken: 0.0000006,
    },
    performance: {
      avgLatencyMs: 800,
      p95LatencyMs: 2000,
    },
  },
  {
    name: "claude-3-opus",
    provider: "anthropic",
    model: "claude-3-opus-20240229",
    capabilities: {
      maxTokens: 200000,
      supportsTools: true,
      supportsVision: true,
      supportsStreaming: true,
    },
    cost: {
      perPromptToken: 0.000015,
      perCompletionToken: 0.000075,
    },
    performance: {
      avgLatencyMs: 3000,
      p95LatencyMs: 8000,
    },
  },
  {
    name: "claude-3-haiku",
    provider: "anthropic",
    model: "claude-3-haiku-20240307",
    capabilities: {
      maxTokens: 200000,
      supportsTools: false,
      supportsVision: true,
      supportsStreaming: true,
    },
    cost: {
      perPromptToken: 0.00000025,
      perCompletionToken: 0.00000125,
    },
    performance: {
      avgLatencyMs: 600,
      p95LatencyMs: 1500,
    },
  },
];
```

## 核心路由器实现

```typescript
// llm-router.ts
interface RouterRequest {
  messages: Array<{ role: string; content: string }>;
  maxTokens?: number;
  temperature?: number;
  tools?: any[];
  images?: string[];
  priority?: "low" | "medium" | "high";
  maxCost?: number;
  modelPreference?: string[];
}

interface RouterResponse {
  model: string;
  provider: string;
  content: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    cost: number;
  };
  latency: number;
  attemptedModels: string[];
}

class LLMRouter {
  private models: Map<string, ModelConfig> = new Map();
  private rules: RoutingRule[] = [];
  private clients: Map<string, any> = new Map();
  private metrics: RouterMetrics;

  constructor() {
    for (const model of MODELS) {
      this.models.set(model.name, model);
    }
    this.metrics = new RouterMetrics();
    this.setupDefaultRules();
  }

  private setupDefaultRules() {
    this.rules = [
      {
        name: "vision 需求",
        priority: 100,
        condition: { type: "requires_vision", value: true },
        targetModel: "gpt-4o",
      },
      {
        name: "工具支持",
        priority: 80,
        condition: { type: "requires_tools", value: true },
        targetModel: "gpt-4o",
      },
      {
        name: "快速响应",
        priority: 60,
        condition: { type: "max_latency", maxMs: 1000 },
        targetModel: "gpt-4o-mini",
      },
      {
        name: "低成本",
        priority: 40,
        condition: { type: "max_cost", maxUsd: 0.001 },
        targetModel: "gpt-4o-mini",
      },
      {
        name: "智能任务",
        priority: 20,
        condition: {
          type: "task_type",
          values: ["coding", "analysis", "reasoning"],
        },
        targetModel: "gpt-4o",
      },
    ];
  }

  // 路由请求到模型
  async route(request: RouterRequest): Promise<RouterResponse> {
    const startTime = Date.now();
    const attemptedModels: string[] = [];

    // 1. 确定候选模型
    const candidates = this.evaluateRules(request);

    // 2. 尝试每个候选
    for (const modelName of candidates) {
      const model = this.models.get(modelName);
      if (!model) continue;

      attemptedModels.push(modelName);

      try {
        const response = await this.callModel(model, request);
        const latency = Date.now() - startTime;

        this.metrics.recordSuccess(modelName, latency, response.usage);

        return {
          ...response,
          latency,
          attemptedModels,
        };
      } catch (error) {
        console.error(`模型 ${modelName} 失败：`, error);
        this.metrics.recordFailure(modelName);
        // 继续尝试下一个
      }
    }

    // 3. 如果所有模型都失败则回退
    throw new Error(
      `所有模型均失败。尝试过的模型：${attemptedModels.join(", ")}`
    );
  }

  // 评估路由规则
  private evaluateRules(request: RouterRequest): string[] {
    // 按优先级排序规则
    const sortedRules = [...this.rules].sort(
      (a, b) => b.priority - a.priority
    );

    const matchedModels = new Map<string, number>();

    for (const rule of sortedRules) {
      if (this.evaluateCondition(rule.condition, request)) {
        const currentScore = matchedModels.get(rule.targetModel) || 0;
        matchedModels.set(
          rule.targetModel,
          currentScore + rule.priority
        );
      }
    }

    // 按总分排序
    return Array.from(matchedModels.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([model]) => model);
  }

  // 评估单个条件
  private evaluateCondition(
    condition: RoutingCondition,
    request: RouterRequest
  ): boolean {
    switch (condition.type) {
      case "requires_tools":
        return condition.value === !!request.tools?.length;
      case "requires_vision":
        return condition.value === !!request.images?.length;
      case "max_latency": {
        const model = this.models.get(request.modelPreference?.[0] || "");
        if (model) {
          return model.performance.avgLatencyMs <= condition.maxMs;
        }
        return false;
      }
      case "max_cost": {
        const estimatedTokens = this.estimateTokens(request);
        return estimatedTokens.cost <= condition.maxUsd;
      }
      case "task_type": {
        const lastMessage =
          request.messages[request.messages.length - 1]?.content || "";
        return condition.values.some((v) =>
          this.detectTaskType(lastMessage, v)
        );
      }
      default:
        return false;
    }
  }

  // 调用模型
  private async callModel(
    model: ModelConfig,
    request: RouterRequest
  ): Promise<Omit<RouterResponse, "latency" | "attemptedModels">> {
    switch (model.provider) {
      case "openai":
        return this.callOpenAI(model, request);
      case "anthropic":
        return this.callAnthropic(model, request);
      default:
        throw new Error(`不支持的提供商：${model.provider}`);
    }
  }

  private async callOpenAI(
    model: ModelConfig,
    request: RouterRequest
  ): Promise<Omit<RouterResponse, "latency" | "attemptedModels">> {
    // 模拟 OpenAI 调用
    const promptTokens = this.estimateTokens(request).prompt;
    const completionTokens = Math.min(
      request.maxTokens || 500,
      model.capabilities.maxTokens
    );

    return {
      model: model.model,
      provider: "openai",
      content: `来自 ${model.name} 的模拟响应`,
      usage: {
        promptTokens,
        completionTokens,
        totalTokens: promptTokens + completionTokens,
        cost:
          promptTokens * model.cost.perPromptToken +
          completionTokens * model.cost.perCompletionToken,
      },
    };
  }

  private async callAnthropic(
    model: ModelConfig,
    request: RouterRequest
  ): Promise<Omit<RouterResponse, "latency" | "attemptedModels">> {
    // 模拟 Anthropic 调用
    return this.callOpenAI(model, request);
  }

  // 估计令牌使用量
  private estimateTokens(request: RouterRequest): {
    prompt: number;
    cost: number;
  } {
    const totalChars = request.messages.reduce(
      (sum, m) => sum + m.content.length,
      0
    );
    const promptTokens = Math.ceil(totalChars / 4);

    // 使用最低成本模型估计
    const minCostModel = Array.from(this.models.values()).reduce(
      (min, m) =>
        m.cost.perPromptToken < min.cost.perPromptToken ? m : min
    );

    return {
      prompt: promptTokens,
      cost: promptTokens * minCostModel.cost.perPromptToken,
    };
  }

  // 简单的任务类型检测
  private detectTaskType(content: string, taskType: string): boolean {
    const patterns: Record<string, RegExp> = {
      coding: /code|function|class|implement|debug|refactor|test/i,
      analysis: /analyze|compare|evaluate|assess|review|summarize/i,
      reasoning: /why|how|explain|reason|think|logic|prove/i,
      creative: /write|create|design|generate|compose|draft/i,
    };

    const pattern = patterns[taskType];
    return pattern ? pattern.test(content) : false;
  }
}
```

## 回退链

```typescript
// fallback-chain.ts
interface FallbackConfig {
  primary: string;
  fallbacks: string[];
  timeout: number;
  retryCount: number;
}

class FallbackChain {
  private router: LLMRouter;
  private chainConfig: FallbackConfig[];

  constructor(router: LLMRouter) {
    this.router = router;
    this.chainConfig = [
      {
        primary: "gpt-4o",
        fallbacks: ["claude-3-opus", "gpt-4o-mini", "claude-3-haiku"],
        timeout: 10000,
        retryCount: 1,
      },
    ];
  }

  async executeWithFallback(request: RouterRequest): Promise<RouterResponse> {
    const config = this.chainConfig[0];
    let lastError: Error | null = null;

    // 尝试主要模型，然后尝试回退
    const modelsToTry = [
      config.primary,
      ...config.fallbacks,
    ];

    for (const modelName of modelsToTry) {
      for (let attempt = 0; attempt <= config.retryCount; attempt++) {
        try {
          const response = await this.tryModelWithTimeout(
            modelName,
            request,
            config.timeout
          );

          if (response) {
            return response;
          }
        } catch (error) {
          lastError = error as Error;
          console.error(
            `尝试 ${modelName}（第 ${attempt + 1} 次）失败：`,
            error
          );
        }
      }
    }

    throw lastError || new Error("回退链中的所有模型均失败");
  }

  private async tryModelWithTimeout(
    modelName: string,
    request: RouterRequest,
    timeoutMs: number
  ): Promise<RouterResponse | null> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const result = await this.router.route({
        ...request,
        modelPreference: [modelName],
      });
      return result;
    } finally {
      clearTimeout(timeoutId);
    }
  }
}
```

## 成本追踪

```typescript
// cost-tracker.ts
interface CostEntry {
  model: string;
  promptTokens: number;
  completionTokens: number;
  cost: number;
  timestamp: Date;
  userId?: string;
  taskType?: string;
}

class CostTracker {
  private costs: CostEntry[] = [];
  private monthlyBudget: number;
  private monthlySpend: number = 0;

  constructor(monthlyBudget: number) {
    this.monthlyBudget = monthlyBudget;
  }

  trackCall(entry: Omit<CostEntry, "timestamp">) {
    const costEntry: CostEntry = {
      ...entry,
      timestamp: new Date(),
    };

    this.costs.push(costEntry);
    this.monthlySpend += entry.cost;

    // 检查预算
    if (this.monthlySpend > this.monthlyBudget) {
      console.error(`警告：已超出月度预算（$${this.monthlySpend} / $${this.monthlyBudget}）`);
    }
  }

  // 按模型汇总成本
  getCostByModel(): Map<string, number> {
    const costs = new Map<string, number>();
    for (const entry of this.costs) {
      costs.set(entry.model, (costs.get(entry.model) || 0) + entry.cost);
    }
    return costs;
  }

  // 获取每日成本
  getDailyCost(days: number = 30): Array<{ date: string; cost: number }> {
    const daily = new Map<string, number>();
    const cutoff = Date.now() - days * 86400000;

    for (const entry of this.costs) {
      if (entry.timestamp.getTime() < cutoff) continue;
      const dateKey = entry.timestamp.toISOString().slice(0, 10);
      daily.set(dateKey, (daily.get(dateKey) || 0) + entry.cost);
    }

    return Array.from(daily.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, cost]) => ({ date, cost }));
  }

  // 估计调用成本
  estimateCost(
    model: string,
    promptTokens: number,
    completionTokens: number
  ): number {
    const modelConfig = MODELS.find((m) => m.name === model);
    if (!modelConfig) return 0;

    return (
      promptTokens * modelConfig.cost.perPromptToken +
      completionTokens * modelConfig.cost.perCompletionToken
    );
  }
}
```

## MCP 集成

```typescript
// mcp-router-integration.ts
class MCPRouterMiddleware {
  private router: LLMRouter;
  private costTracker: CostTracker;

  constructor() {
    this.router = new LLMRouter();
    this.costTracker = new CostTracker(100); // 100 美元月度预算
  }

  // 包装 MCP 工具执行以进行智能路由
  async executeToolWithRouter(
    toolName: string,
    args: unknown,
    context: { userId: string; priority: string }
  ) {
    const startTime = Date.now();

    // 根据工具和上下文确定路由策略
    const request: RouterRequest = {
      messages: [
        {
          role: "user",
          content: `执行工具 ${toolName}，参数：${JSON.stringify(args)}`,
        },
      ],
      tools: [{ name: toolName, parameters: args as Record<string, unknown> }],
      priority: context.priority as "low" | "medium" | "high",
    };

    try {
      const response = await this.router.route(request);

      // 追踪成本
      this.costTracker.trackCall({
        model: response.model,
        promptTokens: response.usage.promptTokens,
        completionTokens: response.usage.completionTokens,
        cost: response.usage.cost,
        userId: context.userId,
        taskType: toolName,
      });

      return response.content;
    } catch (error) {
      console.error(`工具 ${toolName} 执行失败：`, error);
      throw error;
    }
  }
}
```

## 最佳实践

1. **分层路由**：从快速/便宜的模型开始，为复杂任务升级到更昂贵的模型。
2. **监控路由决策**：追踪路由决策以随时间优化规则。
3. **预算告警**：当接近预算限制时设置告警，并自动切换到更便宜的模型。
4. **A/B 测试**：对相似请求使用路由来比较不同模型的输出。
5. **缓存路由决策**：对相似请求模式缓存路由结果以减少开销。

## 练习

1. **基于规则的路由器**：实现一个根据任务类型和复杂度在不同模型间路由的规则引擎。

2. **回退配置**：创建一个具有可配置回退链、超时和重试策略的路由器。

3. **成本仪表盘**：构建一个显示实时和历史的 LLM 使用成本的 Web UI。

4. **智能路由器**：实现一个使用小型/快速 LLM 决定请求应路由到哪个大型模型的路由器。

5. **A/B 测试路由器**：创建一个将一定比例的流量路由到不同模型并比较结果质量的路由器。

## 术语表

- **路由**：基于规则或启发式选择将请求分发到不同模型的过程。
- **回退链**：主要模型失败时尝试的备选模型有序列表。
- **成本感知路由**：基于每个模型的令牌成本优化成本的策略。
- **延迟预算**在为获得可接受的用户体验而必须完成响应的最大时间。
- **模型选择器**：确定哪个模型应处理特定请求的组件。

## 延伸阅读

- LLM 路由策略
- 模型成本对比
- 回退模式（稳定性模式）
- AI 网关架构（Portkey、LiteLLM）
