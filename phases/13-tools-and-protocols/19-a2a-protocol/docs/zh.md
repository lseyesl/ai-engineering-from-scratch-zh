# A2A 协议

> Agent-to-Agent (A2A) 协议使 AI 代理能够直接相互通信、协商和协作完成任务。

**类型：** 构建（Build）
**语言：** TypeScript
**前置要求：** MCP 基础，HTTP 协议知识
**时间：** ~55 分钟

## 学习目标

- 理解 A2A 协议架构及其与 MCP 的关系
- 实现代理发现和功能公告
- 创建代理间通信的消息格式
- 实现任务委托和结果共享
- 构建一个多代理协作系统

## 什么是 A2A 协议？

Agent-to-Agent (A2A) 协议使 AI 代理能够直接相互通信。虽然 MCP 侧重于"主机到代理"的集成，但 A2A 支持"代理到代理"的交互，包括：

- **发现**：找到能够处理特定任务的代理
- **协商**：就任务参数和约束达成一致
- **委托**：将任务移交给另一个代理
- **协调**：同步多个代理的并行操作

```
┌─────────────────────────────────────────────────┐
│                 A2A 生态系统                      │
│                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│  │ 代理 A   │◄──►│ 代理 B   │◄──►│ 代理 C   │    │
│  │ (主代理) │    │ (专家)   │    │ (专家)   │    │
│  └────┬─────┘    └──────────┘    └──────────┘    │
│       │                                            │
│       ▼                                            │
│  ┌──────────────────┐                              │
│  │  A2A 注册中心     │                              │
│  │ (服务发现 + 能力) │                              │
│  └──────────────────┘                              │
└─────────────────────────────────────────────────┘
```

### A2A vs MCP

| 方面 | MCP | A2A |
|------|-----|-----|
| 焦点 | 工具 → 主机 | 代理 → 代理 |
| 拓扑 | 中心化 | 去中心化 |
| 通信 | JSON-RPC | JSON-RPC + 事件 |
| 发现 | 初始化时 | 持续注册 |
| 任务 | 同步 | 同步和异步 |

## A2A 代理规范

```typescript
// a2a-types.ts
// A2A 代理能力声明
interface AgentCapability {
  name: string;                    // 能力名称
  version: string;                 // 能力版本
  description: string;             // 能力描述
  inputSchema: object;             // 接受的输入格式
  outputSchema: object;            // 产生的输出格式
  constraints?: {
    maxInputSize?: number;         // 最大输入大小（字节）
    maxProcessingTime?: number;    // 最大处理时间（毫秒）
    requiresApproval?: boolean;    // 是否需人工审批
    allowedCollaborators?: string[]; // 允许协作者的代理 ID
  };
  cost?: {
    perCall?: number;              // 每次调用的成本
    perToken?: number;             // 每令牌成本
  };
}

// A2A 代理公告
interface AgentAnnouncement {
  agentId: string;                 // 唯一代理 ID
  name: string;                    // 人类可读名称
  version: string;                 // 代理版本
  description: string;             // 代理描述
  endpoint: string;                // 通信端点 URL
  capabilities: AgentCapability[]; // 支持的能力
  status: "available" | "busy" | "away" | "offline";
  metadata: Record<string, unknown>;
}
```

## A2A 通信协议

```typescript
// a2a-protocol.ts
// A2A 消息类型
type A2AMessageType =
  | "DISCOVER"      // 发现代理能力
  | "OFFER"         // 提供执行任务
  | "REQUEST"       // 请求任务执行
  | "ACCEPT"        // 接受任务
  | "REJECT"        // 拒绝任务
  | "PROGRESS"      // 任务进度更新
  | "COMPLETE"      // 任务完成
  | "FAIL"          // 任务失败
  | "CANCEL"        // 取消任务
  | "QUERY"         // 查询信息
  | "RESPOND";      // 响应查询

// A2A 消息格式
interface A2AMessage {
  messageId: string;               // 唯一消息 ID
  type: A2AMessageType;            // 消息类型
  from: string;                    // 发送方代理 ID
  to: string;                      // 接收方代理 ID
  conversationId: string;          // 对话 ID（关联消息）
  timestamp: string;               // ISO 时间戳
  payload: A2APayload;             // 消息负载
  signature?: string;              // 可选数字签名
  ttl?: number;                    // 生存时间（毫秒）
}

// A2A 负载类型
type A2APayload =
  | DiscoveryPayload
  | TaskOfferPayload
  | TaskRequestPayload
  | TaskResultPayload
  | ProgressPayload
  | QueryPayload;

interface DiscoveryPayload {
  type: "discovery";
  requiredCapabilities?: string[];
  context?: Record<string, unknown>;
}

interface TaskOfferPayload {
  type: "task_offer";
  taskId: string;
  capability: string;
  parameters: unknown;
  deadline?: string;
  priority?: "low" | "medium" | "high";
  compensation?: {
    maxCost?: number;
    currency?: string;
  };
}

interface TaskRequestPayload {
  type: "task_request";
  taskId: string;
  capability: string;
  parameters: unknown;
  context?: Record<string, unknown>;
  collaboration?: {
    type: "delegation" | "parallel" | "chain";
    dependsOn?: string[];           // 依赖的任务 ID
  };
}

interface TaskResultPayload {
  type: "task_result";
  taskId: string;
  status: "success" | "failure" | "partial";
  output: unknown;
  error?: string;
  metrics?: {
    processingTime: number;
    tokensUsed?: number;
    cost?: number;
  };
}

interface ProgressPayload {
  type: "progress";
  taskId: string;
  progress: number;                // 0-100
  message?: string;
  estimatedRemaining?: number;     // 预计剩余时间（毫秒）
}
```

## 实现 A2A 代理

```typescript
// a2a-agent.ts
import { EventEmitter } from "events";

class A2AAgent extends EventEmitter {
  protected agentId: string;
  protected name: string;
  protected capabilities: AgentCapability[];
  protected status: AgentAnnouncement["status"];
  protected registry: A2ARegistryClient;
  protected activeConversations: Map<string, A2AConversation>;
  protected pendingTasks: Map<string, TaskRequestPayload>;

  constructor(config: {
    agentId: string;
    name: string;
    capabilities: AgentCapability[];
    registryEndpoint: string;
  }) {
    super();
    this.agentId = config.agentId;
    this.name = config.name;
    this.capabilities = config.capabilities;
    this.status = "available";
    this.registry = new A2ARegistryClient(config.registryEndpoint);
    this.activeConversations = new Map();
    this.pendingTasks = new Map();
  }

  // 公告此代理
  async announce(): Promise<void> {
    const announcement: AgentAnnouncement = {
      agentId: this.agentId,
      name: this.name,
      version: "1.0.0",
      description: `${this.name} 代理`,
      endpoint: `http://localhost:${this.getPort()}/a2a`,
      capabilities: this.capabilities,
      status: this.status,
      metadata: {
        language: "TypeScript",
        framework: "A2A Protocol",
      },
    };

    await this.registry.register(announcement);
    this.startHeartbeat();
  }

  // 处理传入的 A2A 消息
  async handleMessage(message: A2AMessage): Promise<A2AMessage> {
    console.error(
      `[${this.agentId}] 收到来自 ${message.from} 的 ${message.type}`
    );

    switch (message.type) {
      case "DISCOVER":
        return this.handleDiscovery(message);
      case "OFFER":
        return this.handleOffer(message);
      case "REQUEST":
        return this.handleTaskRequest(message);
      case "QUERY":
        return this.handleQuery(message);
      case "PROGRESS":
        return this.handleProgress(message);
      case "COMPLETE":
        return this.handleComplete(message);
      case "CANCEL":
        return this.handleCancel(message);
      default:
        return this.createMessage(
          message.from,
          "RESPOND",
          message.conversationId,
          {
            type: "error",
            error: `不支持的消息类型：${message.type}`,
          }
        );
    }
  }

  // 发现处理程序
  private async handleDiscovery(
    message: A2AMessage
  ): Promise<A2AMessage> {
    const payload = message.payload as DiscoveryPayload;

    // 过滤匹配的能力
    const matchedCapabilities = this.capabilities.filter((cap) => {
      if (!payload.requiredCapabilities) return true;
      return payload.requiredCapabilities.some(
        (req) => cap.name === req || cap.name.includes(req)
      );
    });

    return this.createMessage(
      message.from,
      "RESPOND",
      message.conversationId,
      {
        type: "discovery_result",
        agentId: this.agentId,
        capabilities: matchedCapabilities,
      }
    );
  }

  // 任务报价处理程序
  private async handleOffer(
    message: A2AMessage
  ): Promise<A2AMessage> {
    const offer = message.payload as TaskOfferPayload;

    // 检查是否能处理此能力
    const capability = this.capabilities.find(
      (c) => c.name === offer.capability
    );

    if (!capability) {
      return this.createMessage(
        message.from,
        "REJECT",
        message.conversationId,
        {
          type: "rejection",
          taskId: offer.taskId,
          reason: "不支持的 capability",
        }
      );
    }

    // 检查可用性
    if (this.status === "busy" && this.pendingTasks.size >= 3) {
      return this.createMessage(
        message.from,
        "REJECT",
        message.conversationId,
        {
          type: "rejection",
          taskId: offer.taskId,
          reason: "当前正忙",
        }
      );
    }

    // 接受报价
    this.pendingTasks.set(offer.taskId, {
      type: "task_request",
      taskId: offer.taskId,
      capability: offer.capability,
      parameters: offer.parameters,
    });

    return this.createMessage(
      message.from,
      "ACCEPT",
      message.conversationId,
      {
        type: "acceptance",
        taskId: offer.taskId,
        estimatedCompletion: new Date(
          Date.now() + 30000
        ).toISOString(),
      }
    );
  }

  // 任务执行
  private async handleTaskRequest(
    message: A2AMessage
  ): Promise<A2AMessage> {
    const request = message.payload as TaskRequestPayload;
    const conversation: A2AConversation = {
      id: message.conversationId,
      peerId: message.from,
      taskId: request.taskId,
      startedAt: new Date(),
      status: "running",
    };

    this.activeConversations.set(conversation.id, conversation);
    this.status = "busy";

    // 开始执行任务（异步）
    this.executeTask(request, conversation).catch(console.error);

    // 初始确认
    return this.createMessage(
      message.from,
      "RESPOND",
      message.conversationId,
      {
        type: "acknowledgment",
        taskId: request.taskId,
        status: "accepted",
      }
    );
  }

  // 实际任务执行
  private async executeTask(
    request: TaskRequestPayload,
    conversation: A2AConversation
  ): Promise<void> {
    try {
      // 模拟进度更新
      for (let progress = 0; progress <= 100; progress += 25) {
        await this.sleep(2000);

        const progressMsg = this.createMessage(
          conversation.peerId,
          "PROGRESS",
          conversation.id,
          {
            type: "progress",
            taskId: request.taskId,
            progress,
            message: `进度：${progress}%`,
          }
        );

        await this.sendMessage(progressMsg);
      }

      // 完成任务
      const result = await this.processTask(
        request.capability,
        request.parameters
      );

      const completeMsg = this.createMessage(
        conversation.peerId,
        "COMPLETE",
        conversation.id,
        {
          type: "task_result",
          taskId: request.taskId,
          status: "success",
          output: result,
          metrics: {
            processingTime: 8000,
            tokensUsed: 1500,
          },
        }
      );

      await this.sendMessage(completeMsg);
      conversation.status = "completed";
    } catch (error) {
      const failMsg = this.createMessage(
        conversation.peerId,
        "FAIL",
        conversation.id,
        {
          type: "task_result",
          taskId: request.taskId,
          status: "failure",
          error: String(error),
        }
      );

      await this.sendMessage(failMsg);
      conversation.status = "failed";
    } finally {
      this.status = "available";
      this.pendingTasks.delete(request.taskId);
    }
  }

  // 工具函数
  protected createMessage(
    to: string,
    type: A2AMessageType,
    conversationId: string,
    payload: A2APayload
  ): A2AMessage {
    return {
      messageId: crypto.randomUUID(),
      type,
      from: this.agentId,
      to,
      conversationId,
      timestamp: new Date().toISOString(),
      payload,
    };
  }

  protected async sendMessage(message: A2AMessage): Promise<void> {
    // 通过 HTTP 发送消息到目标代理
    const agent = await this.registry.resolve(message.to);
    if (!agent) throw new Error(`代理 ${message.to} 未找到`);

    const response = await fetch(agent.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message),
    });

    if (!response.ok) {
      throw new Error(`发送消息到 ${agent.name} 失败`);
    }
  }

  protected async processTask(
    capability: string,
    parameters: unknown
  ): Promise<unknown> {
    // 由子类实现
    return { processed: true, capability, parameters };
  }

  protected getPort(): number {
    // 由子类实现
    return 3000;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private startHeartbeat() {
    setInterval(async () => {
      await this.registry.heartbeat(this.agentId, this.status);
    }, 30000);
  }

  private handleQuery(message: A2AMessage): Promise<A2AMessage> {
    throw new Error("方法未实现。");
  }

  private handleProgress(message: A2AMessage): Promise<A2AMessage> {
    throw new Error("方法未实现。");
  }

  private handleComplete(message: A2AMessage): Promise<A2AMessage> {
    throw new Error("方法未实现。");
  }

  private handleCancel(message: A2AMessage): Promise<A2AMessage> {
    throw new Error("方法未实现。");
  }
}

interface A2AConversation {
  id: string;
  peerId: string;
  taskId: string;
  startedAt: Date;
  status: "running" | "completed" | "failed";
}
```

## A2A 注册中心客户端

```typescript
// a2a-registry-client.ts
class A2ARegistryClient {
  private endpoint: string;
  private cache: Map<string, AgentAnnouncement> = new Map();

  constructor(endpoint: string) {
    this.endpoint = endpoint;
  }

  async register(announcement: AgentAnnouncement): Promise<void> {
    const response = await fetch(`${this.endpoint}/agents`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(announcement),
    });

    if (!response.ok) {
      throw new Error(`代理注册失败`);
    }
  }

  async discover(
    requiredCapability: string
  ): Promise<AgentAnnouncement[]> {
    const response = await fetch(
      `${this.endpoint}/discover?capability=${requiredCapability}`
    );

    if (!response.ok) return [];

    const agents = await response.json();
    // 缓存结果
    for (const agent of agents) {
      this.cache.set(agent.agentId, agent);
    }

    return agents;
  }

  async resolve(agentId: string): Promise<AgentAnnouncement | null> {
    // 检查缓存
    if (this.cache.has(agentId)) {
      return this.cache.get(agentId)!;
    }

    // 查询注册中心
    const response = await fetch(
      `${this.endpoint}/agents/${agentId}`
    );

    if (!response.ok) return null;

    const agent = await response.json();
    this.cache.set(agent.agentId, agent);
    return agent;
  }

  async heartbeat(
    agentId: string,
    status: AgentAnnouncement["status"]
  ): Promise<void> {
    await fetch(`${this.endpoint}/agents/${agentId}/heartbeat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    });
  }
}
```

## 主代理示例

```typescript
// orchestrator-agent.ts
class OrchestratorAgent extends A2AAgent {
  private workerCache: Map<string, AgentAnnouncement> = new Map();

  constructor(registryEndpoint: string) {
    super({
      agentId: "orchestrator-1",
      name: "Orchestrator",
      capabilities: [
        {
          name: "task_planning",
          version: "1.0",
          description: "将复杂任务分解为子任务",
          inputSchema: { type: "object" },
          outputSchema: { type: "object" },
        },
        {
          name: "result_aggregation",
          version: "1.0",
          description: "聚合来自多个代理的结果",
          inputSchema: { type: "object" },
          outputSchema: { type: "object" },
        },
      ],
      registryEndpoint,
    });
  }

  // 分解并分发复杂任务
  async orchestrateTask(
    taskDescription: string,
    requirements: string[]
  ): Promise<unknown> {
    console.error(`[Orchestrator] 开始编排任务：${taskDescription}`);

    // 1. 为每个要求发现合适的代理
    const workerPromises = requirements.map((req) =>
      this.registry.discover(req)
    );

    const workersByRequirement = await Promise.all(workerPromises);

    // 2. 将子任务分发给工人
    const subTasks = requirements.map((req, index) => {
      const workers = workersByRequirement[index];
      if (workers.length === 0) {
        throw new Error(`未找到能处理 ${req} 的代理`);
      }

      const worker = workers[0]; // 选择第一个可用代理
      return {
        worker,
        requirement: req,
        taskId: crypto.randomUUID(),
      };
    });

    // 3. 并行委托子任务
    const delegationPromises = subTasks.map((subTask) =>
      this.delegateSubTask(subTask)
    );

    const results = await Promise.all(delegationPromises);

    // 4. 聚合结果
    return this.aggregateResults(taskDescription, results);
  }

  private async delegateSubTask(subTask: {
    worker: AgentAnnouncement;
    requirement: string;
    taskId: string;
  }): Promise<unknown> {
    const offerMessage = this.createMessage(
      subTask.worker.agentId,
      "OFFER",
      crypto.randomUUID(),
      {
        type: "task_offer",
        taskId: subTask.taskId,
        capability: subTask.requirement,
        parameters: {
          description: `子任务：${subTask.requirement}`,
          priority: "high",
        },
      }
    );

    await this.sendMessage(offerMessage);
    // 处理响应……
    return { taskId: subTask.taskId, status: "delegated" };
  }

  private async aggregateResults(
    taskDescription: string,
    results: unknown[]
  ): Promise<unknown> {
    return {
      taskDescription,
      subResults: results,
      aggregatedAt: new Date().toISOString(),
    };
  }
}
```

## 最佳实践

1. **能力公告**：代理应准确描述其能力以避免委托失败。
2. **超时处理**：始终设置消息 TTL 和任务超时。
3. **优雅降级**：当首选代理不可用时回退到替代代理。
4. **错误传播**：清晰地在代理间传播错误上下文。
5. **安全性**：在代理间通信中实现签名和可选加密。

## 练习

1. **简单 A2A 对话**：创建两个直接通信并委托任务的代理。

2. **代理注册中心**：构建一个具有健康检查和能力查询的 A2A 注册中心。

3. **工作流编排器**：实现一个将复杂工作流分解并分发给多个专业代理的主代理。

4. **A2A 安全**：向 A2A 消息添加签名验证以防范中间人攻击。

5. **代理市场**：构建一个用户可以浏览可用代理及其能力的 UI。

## 术语表

- **A2A**：Agent-to-Agent 协议，用于代理间直接通信。
- **能力**：代理可以执行的功能单元。
- **发现**：找到并连接可用代理的过程。
- **委托**：将任务移交给另一个代理执行。
- **编排**：协调多个代理完成复杂工作流。

## 延伸阅读

- A2A 协议规范
- 多代理系统模式
- 服务发现架构
- 分布式系统协调模式
