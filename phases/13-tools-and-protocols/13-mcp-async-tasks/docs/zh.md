# MCP 异步任务

> MCP 异步任务允许服务器长时间运行的操作无需阻塞客户端，支持进度跟踪和结果检索。

**类型：** 构建（Build）
**语言：** TypeScript
**前置要求：** MCP 基础，TypeScript 异步编程
**时间：** ~50 分钟

## 学习目标

- 理解 MCP 中异步任务的需求
- 实现一个支持长时间运行操作的 MCP 服务器
- 使用任务状态管理跟踪进度
- 处理任务取消和错误恢复
- 构建一个实际使用异步任务的文件处理服务器

## 为什么需要异步任务？

MCP 中的工具操作通常是同步的——客户端发送请求，服务器返回响应。但对于以下情况：

- 大型文件处理（视频编码、数据转换）
- 外部 API 调用（Web 爬取、数据获取）
- 复杂计算（模型训练、数据分析）
- 需要用户确认的步骤

...同步方法会阻塞客户端。异步任务提供了一种更好的方式：

```
同步流程（阻塞）：
客户端 ──请求──> 服务器 ──处理中...──> 客户端等待
                 （30秒后）
客户端 <──响应── 服务器 ✅

异步流程（非阻塞）：
客户端 ──请求──> 服务器 ──task_id──> 客户端
客户端 ──轮询──> 服务器 ──进度──> 客户端
                 （30秒后）
客户端 ──获取──> 服务器 ✅
```

## 任务生命周期

```
┌─────────┐
│  待执行   │ ← 任务已创建
└────┬────┘
     │
     ▼
┌─────────┐     ┌──────────┐
│ 运行中   │ ──> │  已完成   │
└────┬────┘     └──────────┘
     │
     ▼
┌─────────┐
│  失败    │
└─────────┘
```

## 任务 API

```typescript
// 任务状态
type TaskStatus = "pending" | "running" | "completed" | "failed";

// 任务信息
interface Task {
  id: string;                    // 唯一任务标识符
  status: TaskStatus;            // 当前状态
  progress?: number;             // 进度百分比（0-100）
  createdAt: string;             // ISO 时间戳
  completedAt?: string;          // 完成时间戳
  result?: unknown;              // 任务结果（完成后）
  error?: string;                // 错误信息（失败时）
}

// 创建异步任务
interface TaskCreateRequest {
  method: "tools/call";
  params: {
    name: string;        // 工具名称
    arguments?: object;  // 工具参数
  };
}

// 任务状态响应
interface TaskStatusResponse {
  _meta?: {
    progress?: number;       // 进度 0-100
    message?: string;        // 当前状态消息
  };
  content: ToolContent[];    // 完成后的结果
  isError?: boolean;
}
```

## 实现异步任务支持

让我们构建一个支持异步任务的 MCP 服务器：

```typescript
// async-task-server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { randomUUID } from "crypto";

// 任务管理器
class TaskManager {
  private tasks: Map<string, Task> = new Map();
  private handlers: Map<string, TaskHandler> = new Map();

  // 注册任务处理程序
  register(name: string, handler: TaskHandler) {
    this.handlers.set(name, handler);
  }

  // 创建异步任务
  async createTask(toolName: string, args: unknown): Promise<Task> {
    const handler = this.handlers.get(toolName);
    if (!handler) {
      throw new Error(`未找到工具：${toolName}`);
    }

    const task: Task = {
      id: randomUUID(),
      status: "pending",
      progress: 0,
      createdAt: new Date().toISOString(),
    };

    this.tasks.set(task.id, task);

    // 在后台启动任务
    this.executeTask(task, handler, args).catch(console.error);

    return task;
  }

  // 获取任务状态
  getTask(taskId: string): Task | undefined {
    return this.tasks.get(taskId);
  }

  // 列出所有任务
  listTasks(status?: TaskStatus): Task[] {
    const all = Array.from(this.tasks.values());
    return status ? all.filter(t => t.status === status) : all;
  }

  // 取消任务
  async cancelTask(taskId: string): Promise<boolean> {
    const task = this.tasks.get(taskId);
    if (!task || task.status === "completed") {
      return false;
    }

    task.status = "failed";
    task.error = "用户取消";
    task.completedAt = new Date().toISOString();
    return true;
  }

  // 在后台执行任务
  private async executeTask(
    task: Task,
    handler: TaskHandler,
    args: unknown
  ) {
    try {
      task.status = "running";

      const result = await handler.execute(args, (progress, message) => {
        task.progress = progress;
        console.error(`进度 [${task.id}]: ${progress}% - ${message}`);
      });

      task.status = "completed";
      task.progress = 100;
      task.result = result;
      task.completedAt = new Date().toISOString();
    } catch (error) {
      task.status = "failed";
      task.error = error instanceof Error ? error.message : "未知错误";
      task.completedAt = new Date().toISOString();
    }
  }
}

// 任务处理程序接口
interface TaskHandler {
  execute(
    args: unknown,
    onProgress: (progress: number, message: string) => void
  ): Promise<unknown>;
}

// 示例：大型文件处理程序
class FileProcessor implements TaskHandler {
  async execute(
    args: unknown,
    onProgress: (progress: number, message: string) => void
  ): Promise<unknown> {
    const { filePath, operation } = args as {
      filePath: string;
      operation: string;
    };

    // 模拟处理步骤
    const steps = [
      { progress: 10, message: "读取文件..." },
      { progress: 30, message: "处理中..." },
      { progress: 50, message: "转换中..." },
      { progress: 70, message: "验证结果..." },
      { progress: 90, message: "写入输出..." },
    ];

    for (const step of steps) {
      await new Promise(resolve => setTimeout(resolve, 2000)); // 模拟工作
      onProgress(step.progress, step.message);
    }

    return {
      success: true,
      filePath,
      operation,
      outputFile: `${filePath}.processed`,
      size: "2.3 MB",
      duration: "8.5 秒",
    };
  }
}

// 设置服务器
const taskManager = new TaskManager();
taskManager.register("process_file", new FileProcessor());

const server = new Server(
  {
    name: "async-task-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// 列出工具
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "process_file",
      description: "异步处理一个文件。返回 task_id 用于跟踪。",
      inputSchema: {
        type: "object",
        properties: {
          filePath: {
            type: "string",
            description: "要处理的文件路径",
          },
          operation: {
            type: "string",
            description: "要执行的操作",
            enum: ["compress", "convert", "analyze"],
          },
        },
        required: ["filePath", "operation"],
      },
    },
    {
      name: "get_task_status",
      description: "获取异步任务的状态",
      inputSchema: {
        type: "object",
        properties: {
          taskId: {
            type: "string",
            description: "任务 ID",
          },
        },
        required: ["taskId"],
      },
    },
    {
      name: "list_tasks",
      description: "列出所有任务（可选的按状态过滤）",
      inputSchema: {
        type: "object",
        properties: {
          status: {
            type: "string",
            description: "过滤状态",
            enum: ["pending", "running", "completed", "failed"],
          },
        },
      },
    },
    {
      name: "cancel_task",
      description: "取消一个运行中的任务",
      inputSchema: {
        type: "object",
        properties: {
          taskId: {
            type: "string",
            description: "要取消的任务 ID",
          },
        },
        required: ["taskId"],
      },
    },
  ],
}));

// 处理工具调用
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "process_file": {
      const task = await taskManager.createTask(name, args);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                message: "任务已创建",
                taskId: task.id,
                status: task.status,
              },
              null,
              2
            ),
          },
        ],
      };
    }

    case "get_task_status": {
      const taskId = String((args as { taskId: string }).taskId);
      const task = taskManager.getTask(taskId);

      if (!task) {
        throw new Error(`任务未找到：${taskId}`);
      }

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(task, null, 2),
          },
        ],
      };
    }

    case "list_tasks": {
      const status = (args as { status?: TaskStatus })?.status;
      const tasks = taskManager.listTasks(status);

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({ tasks, count: tasks.length }, null, 2),
          },
        ],
      };
    }

    case "cancel_task": {
      const taskId = String((args as { taskId: string }).taskId);
      const cancelled = await taskManager.cancelTask(taskId);

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              { cancelled, taskId },
              null,
              2
            ),
          },
        ],
      };
    }

    default:
      throw new Error(`未知工具：${name}`);
  }
});

// 启动
const transport = new StdioServerTransport();
await server.connect(transport);
```

## 高级：WebSocket 进度推送

除了轮询，还可以使用 WebSocket 推送进度更新：

```typescript
class WebSocketTaskManager extends TaskManager {
  private clients: Set<WebSocket> = new Set();

  registerClient(ws: WebSocket) {
    this.clients.add(ws);
    ws.on("close", () => this.clients.delete(ws));
  }

  protected async executeTask(
    task: Task,
    handler: TaskHandler,
    args: unknown
  ) {
    // 重写进度处理以广播更新
    const originalExecute = handler.execute.bind(handler);

    task.status = "running";

    try {
      const result = await handler.execute(args, (progress, message) => {
        task.progress = progress;
        this.broadcast({
          type: "task_progress",
          taskId: task.id,
          progress,
          message,
        });
      });

      task.status = "completed";
      task.result = result;
      this.broadcast({
        type: "task_completed",
        taskId: task.id,
        result,
      });
    } catch (error) {
      task.status = "failed";
      task.error = error instanceof Error ? error.message : "未知错误";
      this.broadcast({
        type: "task_failed",
        taskId: task.id,
        error: task.error,
      });
    }
  }

  private broadcast(data: unknown) {
    const message = JSON.stringify(data);
    for (const client of this.clients) {
      client.send(message);
    }
  }
}
```

## 最佳实践

1. **立即响应**：创建任务后立即返回 `taskId`。
2. **持久化状态**：对任务状态使用持久化存储，以便在服务器重启后恢复。
3. **合理的进度更新**：避免过于频繁的更新——每 5-10% 更新一次即可。
4. **超时处理**：对长时间运行的任务实现超时机制。
5. **清理机制**：定期清理旧任务记录以防止内存泄漏。

## 练习

1. **视频处理服务器**：创建一个使用异步任务处理视频文件（压缩、转码）的服务器。

2. **Web 爬虫工具**：实现一个异步爬取网站并返回结构化数据的工具。

3. **批量数据处理**：构建一个处理大型 CSV/JSON 文件并将行转换为结构化数据的工具。

4. **任务持久化**：添加 SQLite 支持以在服务器重启后保留任务状态。

5. **速率限制**：实现一个限制并行异步任务数量的节流系统。

## 术语表

- **异步任务**：在后台执行并允许客户端继续其他操作的操作。
- **任务 ID**：分配给每个异步任务的唯一标识符。
- **任务状态**：任务当前的生命周期阶段（待执行、运行中、已完成、失败）。
- **进度**：表示任务完成百分比的 0-100 之间的值。
- **轮询**：定期检查状态更新的机制。

## 延伸阅读

- MCP 任务规范
- Node.js 异步编程模式
- WebSocket 实时更新
- 任务队列架构模式
