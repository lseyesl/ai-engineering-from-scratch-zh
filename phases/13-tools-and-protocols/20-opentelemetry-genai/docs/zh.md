# 生成式 AI 的可观测性

> 使用 OpenTelemetry 为生成式 AI 应用实现全面的可观测性，涵盖 LLM 调用追踪、语义缓存、性能监控和调试。

**类型：** 构建（Build）
**语言：** TypeScript
**前置要求：** 可观测性概念基础
**时间：** ~45 分钟

## 学习目标

- 理解 LLM 特定可观测性的挑战
- 实现使用 OpenTelemetry 的 LLM 调用追踪
- 为工具和代理创建语义约定跨度
- 构建 LLM 调用和令牌使用的监控仪表盘
- 诊断和优化 LLM 驱动应用的性能

## 为什么生成式 AI 需要不同的可观测性？

传统可观测性侧重于请求/响应追踪。对于 LLM 应用，我们需要追踪：

1. **提示词和补全**：发送和接收的内容
2. **令牌使用**：成本和速率限制
3. **工具调用**：LLM 发起的工具执行
4. **模型选择**：使用了哪个模型以及原因
5. **缓存命中**：语义缓存是否服务于请求
6. **回退**：是否使用了回退模型

```
用户请求
    │
    ▼
┌─────────────────────────────┐
│     LLM 调用跨度              │
│  ~~~~~~~~~~~~~~~~~~~~~~~~~~  │
│  prompt_tokens: 150         │
│  completion_tokens: 50      │
│  model: gpt-4o              │
│  temperature: 0.7           │
└────────────┬────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌──────────┐    ┌──────────┐
│工具调用跨度│    │嵌入跨度  │
│~~~~~~~~~~~│    │~~~~~~~~~~│
│tool: search│   │model: ada│
│duration: 2s│   │dimensions:1536│
└──────────┘    └──────────┘
```

## OpenTelemetry 语义约定

OpenTelemetry 社区定义了 LLM 使用的语义约定：

```typescript
// llm-semantic-conventions.ts
// LLM 跨度属性键
const LLM_SEMANTICS = {
  // 基本 LLM 调用属性
  SYSTEM: "gen_ai.system",           // "openai", "anthropic", "ollama"
  REQUEST_MODEL: "gen_ai.request.model",           // "gpt-4o"
  RESPONSE_MODEL: "gen_ai.response.model",          // "gpt-4o-2024-05-13"
  REQUEST_MAX_TOKENS: "gen_ai.request.max_tokens",  // 1000
  REQUEST_TEMPERATURE: "gen_ai.request.temperature", // 0.7
  REQUEST_TOP_P: "gen_ai.request.top_p",             // 0.9
  REQUEST_FREQUENCY_PENALTY: "gen_ai.request.frequency_penalty",
  REQUEST_PRESENCE_PENALTY: "gen_ai.request.presence_penalty",

  // 令牌使用
  USAGE_PROMPT_TOKENS: "gen_ai.usage.prompt_tokens",
  USAGE_COMPLETION_TOKENS: "gen_ai.usage.completion_tokens",
  USAGE_TOTAL_TOKENS: "gen_ai.usage.total_tokens",

  // 响应
  RESPONSE_FINISH_REASON: "gen_ai.response.finish_reason", // "stop", "length"

  // 工具使用
  TOOL_NAME: "gen_ai.tool.name",
  TOOL_CALL_ID: "gen_ai.tool.call_id",

  // 内容安全
  CONTENT_FLAGGED: "gen_ai.content.flagged",
  CONTENT_CATEGORIES: "gen_ai.content.categories",
} as const;
```

## 实现 LLM 调用追踪

```typescript
// llm-tracer.ts
import { trace, Span, SpanStatusCode, context } from "@opentelemetry/api";
import { Resource } from "@opentelemetry/resources";
import {
  SimpleSpanProcessor,
  ConsoleSpanExporter,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";

// LLM 追踪器类
class LLMTracer {
  private tracer;
  private provider: NodeTracerProvider;

  constructor(serviceName: string) {
    // 初始化 OpenTelemetry
    this.provider = new NodeTracerProvider({
      resource: new Resource({
        "service.name": serviceName,
        "service.version": "1.0.0",
      }),
    });

    // 配置导出器
    this.provider.addSpanProcessor(
      new SimpleSpanProcessor(new ConsoleSpanExporter())
    );

    // 在生产环境中启用 OTLP
    if (process.env.OTEL_EXPORTER_OTLP_ENDPOINT) {
      this.provider.addSpanProcessor(
        new SimpleSpanProcessor(
          new OTLPTraceExporter({
            url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT,
          })
        )
      );
    }

    this.provider.register();
    this.tracer = trace.getTracer(serviceName);
  }

  // 追踪 LLM 调用
  async traceLLMCall<T>(
    callConfig: {
      model: string;
      system?: string;
      messages: Array<{ role: string; content: string }>;
      maxTokens?: number;
      temperature?: number;
    },
    fn: () => Promise<T>
  ): Promise<{
    result: T;
    span: Span;
  }> {
    const span = this.tracer.startSpan("llm.call", {
      attributes: {
        [LLM_SEMANTICS.SYSTEM]: "openai",
        [LLM_SEMANTICS.REQUEST_MODEL]: callConfig.model,
        [LLM_SEMANTICS.REQUEST_MAX_TOKENS]: callConfig.maxTokens || 1000,
        [LLM_SEMANTICS.REQUEST_TEMPERATURE]: callConfig.temperature || 0.7,
        "llm.prompt.system": callConfig.system || "",
        "llm.prompt.messages": JSON.stringify(callConfig.messages.slice(0, 2)),
      },
    });

    const ctx = trace.setSpan(context.active(), span);

    try {
      const result = await context.with(ctx, fn);

      // 设置响应属性
      span.setAttribute(LLM_SEMANTICS.RESPONSE_MODEL, callConfig.model);
      span.setStatus({ code: SpanStatusCode.OK });

      return { result, span };
    } catch (error) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: String(error),
      });
      span.recordException(error as Error);
      throw error;
    } finally {
      span.end();
    }
  }

  // 创建工具调用子跨度
  createToolSpan(toolName: string, args: unknown): Span {
    const span = this.tracer.startSpan(`llm.tool.${toolName}`, {
      attributes: {
        [LLM_SEMANTICS.TOOL_NAME]: toolName,
        [LLM_SEMANTICS.TOOL_CALL_ID]: crypto.randomUUID(),
        "tool.args": JSON.stringify(args),
      },
    });

    return span;
  }

  // 完成工具跨度
  finishToolSpan(span: Span, result: unknown, error?: Error) {
    if (error) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
      span.recordException(error);
    } else {
      span.setAttribute("tool.result", JSON.stringify(result).slice(0, 1000));
      span.setStatus({ code: SpanStatusCode.OK });
    }
    span.end();
  }

  // 追踪嵌入调用
  async traceEmbedding<T>(
    config: { model: string; input: string[] },
    fn: () => Promise<T>
  ): Promise<T> {
    const span = this.tracer.startSpan("llm.embedding", {
      attributes: {
        [LLM_SEMANTICS.SYSTEM]: "openai",
        [LLM_SEMANTICS.REQUEST_MODEL]: config.model,
        "embedding.input_count": config.input.length,
        "embedding.input_size": config.input.reduce((s, i) => s + i.length, 0),
      },
    });

    try {
      const result = await fn();
      span.setStatus({ code: SpanStatusCode.OK });
      return result;
    } catch (error) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: String(error) });
      throw error;
    } finally {
      span.end();
    }
  }

  // 关闭追踪器
  async shutdown() {
    await this.provider.shutdown();
  }
}
```

## 包装 LLM 客户端

```typescript
// instrumented-llm-client.ts
import OpenAI from "openai";

class InstrumentedLLMClient {
  private openai: OpenAI;
  private tracer: LLMTracer;

  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    this.tracer = new LLMTracer("llm-service");
  }

  // 带追踪的聊天补全
  async chat(config: {
    model: string;
    messages: Array<{ role: string; content: string }>;
    maxTokens?: number;
    temperature?: number;
    tools?: any[];
  }) {
    const { result, span } = await this.tracer.traceLLMCall(
      config,
      async () => {
        const startTime = Date.now();

        const completion = await this.openai.chat.completions.create({
          model: config.model,
          messages: config.messages,
          max_tokens: config.maxTokens,
          temperature: config.temperature,
          tools: config.tools,
        });

        const duration = Date.now() - startTime;
        const choice = completion.choices[0];

        // 记录令牌使用
        if (completion.usage) {
          span.setAttribute(LLM_SEMANTICS.USAGE_PROMPT_TOKENS, completion.usage.prompt_tokens);
          span.setAttribute(LLM_SEMANTICS.USAGE_COMPLETION_TOKENS, completion.usage.completion_tokens);
          span.setAttribute(LLM_SEMANTICS.USAGE_TOTAL_TOKENS, completion.usage.total_tokens);
        }

        // 记录完成原因
        span.setAttribute(LLM_SEMANTICS.RESPONSE_FINISH_REASON, choice.finish_reason || "unknown");

        // 记录持续时间
        span.setAttribute("llm.duration_ms", duration);

        // 追踪工具调用
        if (choice.message.tool_calls) {
          for (const toolCall of choice.message.tool_calls) {
            const toolSpan = this.tracer.createToolSpan(
              toolCall.function.name,
              JSON.parse(toolCall.function.arguments)
            );
            span.addLink({
              context: toolSpan.spanContext(),
              attributes: {
                [LLM_SEMANTICS.TOOL_CALL_ID]: toolCall.id,
              },
            });
            toolSpan.end();
          }
        }

        return completion;
      }
    );

    return result;
  }
}
```

## 语义缓存与可观测性

```typescript
// semantic-cache.ts
import { trace, SpanStatusCode } from "@opentelemetry/api";

interface CacheEntry {
  response: unknown;
  embedding: number[];
  timestamp: Date;
  hits: number;
}

class ObservableSemanticCache {
  private cache: Map<string, CacheEntry> = new Map();
  private tracer;

  constructor() {
    const tracerProvider = trace.getTracerProvider();
    this.tracer = tracerProvider.getTracer("semantic-cache");
  }

  async get(query: string): Promise<unknown | null> {
    const span = this.tracer.startSpan("cache.lookup", {
      attributes: {
        "cache.query_length": query.length,
        "cache.size": this.cache.size,
      },
    });

    try {
      const queryEmbedding = await this.getEmbedding(query);
      let bestMatch: { key: string; similarity: number } | null = null;

      for (const [key, entry] of this.cache.entries()) {
        const similarity = this.cosineSimilarity(
          queryEmbedding,
          entry.embedding
        );

        span.addEvent("cache.comparison", {
          key,
          similarity,
          entry_hits: entry.hits,
          entry_age_ms: Date.now() - entry.timestamp.getTime(),
        });

        if (similarity > 0.95 && (!bestMatch || similarity > bestMatch.similarity)) {
          bestMatch = { key, similarity };
        }
      }

      if (bestMatch) {
        const entry = this.cache.get(bestMatch.key)!;
        entry.hits++;

        span.setAttribute("cache.hit", true);
        span.setAttribute("cache.similarity", bestMatch.similarity);
        span.setAttribute("cache.entry_hits", entry.hits);
        span.setStatus({ code: SpanStatusCode.OK });

        return entry.response;
      }

      span.setAttribute("cache.hit", false);
      span.setStatus({ code: SpanStatusCode.OK });
      return null;
    } catch (error) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: String(error) });
      return null;
    } finally {
      span.end();
    }
  }

  async set(query: string, response: unknown): Promise<void> {
    const span = this.tracer.startSpan("cache.store", {
      attributes: {
        "cache.query_length": query.length,
        "cache.response_size": JSON.stringify(response).length,
      },
    });

    try {
      const embedding = await this.getEmbedding(query);

      this.cache.set(query, {
        response,
        embedding,
        timestamp: new Date(),
        hits: 0,
      });

      span.setAttribute("cache.new_size", this.cache.size);
      span.setStatus({ code: SpanStatusCode.OK });
    } catch (error) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: String(error) });
    } finally {
      span.end();
    }
  }

  private async getEmbedding(text: string): Promise<number[]> {
    // 实际嵌入调用将在此处完成
    return new Array(1536).fill(0).map(() => Math.random());
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dot / (magA * magB);
  }
}
```

## 指标收集

```typescript
// llm-metrics.ts
import { Meter, Counter, Histogram } from "@opentelemetry/api";
import { metrics } from "@opentelemetry/api";

class LLMMetricsCollector {
  private meter: Meter;

  // 计数器
  private llmCallsTotal: Counter;
  private toolCallsTotal: Counter;
  private cacheHitsTotal: Counter;
  private errorTotal: Counter;

  // 直方图
  private llmDuration: Histogram;
  private tokenUsage: Histogram;
  private cacheSimilarity: Histogram;

  constructor() {
    this.meter = metrics.getMeter("llm-metrics");

    // 初始化指标
    this.llmCallsTotal = this.meter.createCounter("llm.calls.total", {
      description: "LLM 调用总数",
    });

    this.toolCallsTotal = this.meter.createCounter("llm.tool.calls.total", {
      description: "工具调用总数",
    });

    this.cacheHitsTotal = this.meter.createCounter("llm.cache.hits.total", {
      description: "语义缓存命中总数",
    });

    this.errorTotal = this.meter.createCounter("llm.errors.total", {
      description: "LLM 错误总数",
    });

    this.llmDuration = this.meter.createHistogram("llm.duration.ms", {
      description: "LLM 调用持续时间（毫秒）",
      unit: "ms",
    });

    this.tokenUsage = this.meter.createHistogram("llm.token.usage", {
      description: "每次调用的令牌使用量",
    });

    this.cacheSimilarity = this.meter.createHistogram("llm.cache.similarity", {
      description: "缓存命中相似度分数",
    });
  }

  // 记录 LLM 调用
  recordLLMCall(attrs: {
    model: string;
    duration: number;
    promptTokens: number;
    completionTokens: number;
    success: boolean;
  }) {
    this.llmCallsTotal.add(1, {
      model: attrs.model,
      success: String(attrs.success),
    });

    this.llmDuration.record(attrs.duration, {
      model: attrs.model,
    });

    this.tokenUsage.record(attrs.promptTokens, {
      type: "prompt",
      model: attrs.model,
    });

    this.tokenUsage.record(attrs.completionTokens, {
      type: "completion",
      model: attrs.model,
    });

    if (!attrs.success) {
      this.errorTotal.add(1, { model: attrs.model, type: "llm_error" });
    }
  }

  // 记录工具调用
  recordToolCall(attrs: { toolName: string; duration: number; success: boolean }) {
    this.toolCallsTotal.add(1, {
      tool: attrs.toolName,
      success: String(attrs.success),
    });
  }

  // 记录缓存命中
  recordCacheHit(attrs: { similarity: number }) {
    this.cacheHitsTotal.add(1);
    this.cacheSimilarity.record(attrs.similarity);
  }
}
```

## 性能仪表盘查询

```typescript
// dashboard-queries.ts
// 示例 PromQL/OpenTelemetry 查询用于 LLM 监控

const DASHBOARD_QUERIES = {
  // LLM 调用率（按模型）
  llmCallRate: `
    rate(llm_calls_total[5m])
  `,

  // 按模型的 p95 延迟
  llmLatencyP95: `
    histogram_quantile(0.95,
      rate(llm_duration_ms_bucket[5m])
    )
  `,

  // 令牌使用率
  tokenUsageRate: `
    rate(llm_token_usage_sum[5m])
  `,

  // 错误率
  errorRate: `
    rate(llm_errors_total[5m])
  `,

  // 缓存命中率
  cacheHitRate: `
    rate(llm_cache_hits_total[5m])
  `,

  // 工具调用分布
  toolCallDistribution: `
    sum by(tool) (rate(llm_tool_calls_total[5m]))
  `,
};
```

## 最佳实践

1. **跨度丰富化**：始终向 LLM 跨度添加提示词、模型、令牌和延迟属性。
2. **用户追踪**：使用 `enduser.id` 将 LLM 调用关联到特定用户或会话。
3. **预算追踪**：根据令牌使用监控和预测成本。
4. **异常检测**：设置延迟、错误率和令牌使用的告警阈值。
5. **采样策略**：对高容量 LLM 应用使用头部采样来管理数据量。

## 练习

1. **基本 LLM 追踪**：为 LLM 聊天客户端实现 OpenTelemetry 追踪。

2. **工具调用跨度**：创建一个追踪 LLM 发起的每个工具执行的仪器化工具执行器。

3. **成本追踪仪表盘**：按用户、模型和功能构建追踪和显示 LLM 使用成本的系统。

4. **自定义指标导出器**：创建一个为 LLM 特定指标（如逐提示词令牌成本）导出指标的自定义导出器。

5. **告警集成**：设置基于 LLM 可观测性指标（高延迟、高错误率、令牌使用异常）的告警。

## 术语表

- **跨度**：代表完成特定操作的工作单元。
- **语义约定**：跨不同系统命名和结构化可观测性数据的标准。
- **指标**：代表随时间聚合测量的数字数据点。
- **追踪**：通过系统传播请求时创建的跨度集合。
- **导出器**：将遥测数据发送到可观测性后端（如 Jaeger、Prometheus）的组件。

## 延伸阅读

- OpenTelemetry LLM 语义约定
- OpenTelemetry 文档
- Prometheus 指标最佳实践
- LLM 成本监控指南
