# MCP 网关与注册中心

> MCP 网关和注册中心实现了大规模 MCP 生态系统的服务发现、负载均衡和集中管理。

**类型：** 构建（Build）
**语言：** TypeScript
**前置要求：** MCP 基础，HTTP 服务概念
**时间：** ~55 分钟

## 学习目标

- 理解 MCP 网关架构及其组件
- 实现 MCP 注册中心以进行服务发现
- 构建跨多个 MCP 服务器的负载均衡
- 实现网关级缓存和速率限制
- 设计 MCP 服务器的健康检查和监控

## 为什么需要 MCP 网关？

随着 MCP 生态系统的增长，管理多个服务器变得有挑战性。MCP 网关提供：

1. **统一端点**：所有 MCP 服务器的单一入口点
2. **服务发现**：自动检测可用的 MCP 服务器
3. **负载均衡**：跨服务器分发请求
4. **集中策略**：全局速率限制、认证和监控

```
┌─────────┐
│ 客户端   │
└────┬────┘
     │
     ▼
┌─────────────────────────────┐
│       MCP 网关              │
│                             │
│  ┌─────────┐ ┌───────────┐  │
│  │路由器   │ │速率限制器 │  │
│  ├─────────┤ ├───────────┤  │
│  │认证    │ │缓存      │  │
│  └─────────┘ └───────────┘  │
└────────┬──────────┬─────────┘
         │          │
         ▼          ▼
┌──────────┐ ┌──────────┐
│MCP 服务器1│ │MCP 服务器2│
└──────────┘ └──────────┘
     │              │
     ▼              ▼
┌──────────────────────────┐
│    MCP 注册中心          │
│  (服务发现 + 健康检查)    │
└──────────────────────────┘
```

## MCP 注册中心

注册中心是 MCP 服务器的中央目录：

```typescript
// registry.ts
interface MCPServerRegistration {
  id: string;
  name: string;
  version: string;
  endpoint: string;
  transport: "stdio" | "sse" | "streamable-http";
  capabilities: {
    tools?: boolean;
    resources?: boolean;
    prompts?: boolean;
    sampling?: boolean;
  };
  metadata: {
    tools: number;
    resources: number;
    prompts: number;
  };
  health: {
    status: "healthy" | "degraded" | "unhealthy";
    lastCheck: Date;
    uptime: number;
    responseTime: number;
  };
}

class MCPRegistry {
  private servers: Map<string, MCPServerRegistration> = new Map();
  private healthCheckTimers: Map<string, NodeJS.Timeout> = new Map();

  // 注册 MCP 服务器
  register(registration: MCPServerRegistration) {
    this.servers.set(registration.id, {
      ...registration,
      health: {
        status: "healthy",
        lastCheck: new Date(),
        uptime: 0,
        responseTime: 0,
      },
    });

    // 开始健康检查
    this.startHealthChecks(registration.id);
  }

  // 从注册中心中注销
  deregister(serverId: string) {
    this.servers.delete(serverId);
    this.stopHealthChecks(serverId);
  }

  // 按能力查找服务器
  findServers(capability: keyof MCPServerRegistration["capabilities"]) {
    return Array.from(this.servers.values()).filter(
      (s) =>
        s.capabilities[capability] &&
        s.health.status === "healthy"
    );
  }

  // 获取健康服务器
  getHealthyServers(): MCPServerRegistration[] {
    return Array.from(this.servers.values()).filter(
      (s) => s.health.status === "healthy"
    );
  }

  // 更新服务器健康状态
  updateHealth(
    serverId: string,
    status: MCPServerRegistration["health"]["status"],
    responseTime: number
  ) {
    const server = this.servers.get(serverId);
    if (server) {
      server.health.status = status;
      server.health.lastCheck = new Date();
      server.health.responseTime = responseTime;
      server.health.uptime = status === "healthy"
        ? server.health.uptime + 1
        : 0;
    }
  }

  // 健康检查循环
  private async startHealthChecks(serverId: string) {
    const timer = setInterval(async () => {
      const server = this.servers.get(serverId);
      if (!server) return;

      try {
        const start = Date.now();
        const response = await fetch(`${server.endpoint}/health`);
        const responseTime = Date.now() - start;

        if (response.ok) {
          this.updateHealth(serverId, "healthy", responseTime);
        } else {
          this.updateHealth(serverId, "degraded", responseTime);
        }
      } catch {
        this.updateHealth(serverId, "unhealthy", 0);
      }
    }, 30_000); // 每 30 秒检查一次

    this.healthCheckTimers.set(serverId, timer);
  }

  private stopHealthChecks(serverId: string) {
    const timer = this.healthCheckTimers.get(serverId);
    if (timer) {
      clearInterval(timer);
      this.healthCheckTimers.delete(serverId);
    }
  }

  // 列出所有已注册的服务器
  listServers(): MCPServerRegistration[] {
    return Array.from(this.servers.values());
  }
}
```

## MCP 网关实现

```typescript
// mcp-gateway.ts
import express from "express";
import { MCPRegistry } from "./registry";
import { MCPCache } from "./cache";
import { RateLimiter } from "./rate-limiter";

class MCPGateway {
  private registry: MCPRegistry;
  private cache: MCPCache;
  private rateLimiter: RateLimiter;
  private app: express.Application;

  constructor() {
    this.registry = new MCPRegistry();
    this.cache = new MCPCache();
    this.rateLimiter = new RateLimiter({
      windowMs: 60_000,
      maxRequests: 100,
    });

    this.app = express();
    this.app.use(express.json());
    this.setupMiddleware();
    this.setupRoutes();
  }

  private setupMiddleware() {
    // 全局速率限制
    this.app.use(async (req, res, next) => {
      const clientIp = req.ip || "unknown";
      const allowed = await this.rateLimiter.checkLimit(clientIp);

      if (!allowed) {
        return res.status(429).json({
          error: "too_many_requests",
          retryAfter: this.rateLimiter.getRetryAfter(clientIp),
        });
      }
      next();
    });

    // 认证中间件
    this.app.use(async (req, res, next) => {
      // 在这里实现认证
      // 可以检查 JWT、API 密钥等
      next();
    });

    // 请求日志
    this.app.use((req, res, next) => {
      console.error(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
      next();
    });
  }

  private setupRoutes() {
    // 发现端点——列出可用服务器
    this.app.get("/discover", (req, res) => {
      const servers = this.registry.getHealthyServers();
      res.json({
        servers: servers.map((s) => ({
          id: s.id,
          name: s.name,
          capabilities: s.capabilities,
          metadata: s.metadata,
        })),
      });
    });

    // 工具路由——分发到合适的服务器
    this.app.post("/tools/call", async (req, res) => {
      const { name, arguments: args } = req.body;

      // 找到能处理此工具的服务器
      const servers = this.registry.findServers("tools");
      if (servers.length === 0) {
        return res.status(503).json({ error: "no_healthy_servers" });
      }

      // 检查缓存
      const cacheKey = `tool:${name}:${JSON.stringify(args)}`;
      const cached = await this.cache.get(cacheKey);
      if (cached) {
        return res.json(cached);
      }

      // 负载均衡——选择服务器
      const server = this.selectServer(servers);

      try {
        const result = await this.proxyRequest(server, {
          method: "tools/call",
          params: { name, arguments: args },
        });

        // 缓存结果（如果可缓存）
        await this.cache.set(cacheKey, result, 60_000);

        res.json(result);
      } catch (error) {
        // 故障转移——尝试下一个服务器
        const fallbackServer = this.selectServer(
          servers.filter((s) => s.id !== server.id)
        );

        if (fallbackServer) {
          const result = await this.proxyRequest(fallbackServer, {
            method: "tools/call",
            params: { name, arguments: args },
          });
          return res.json(result);
        }

        res.status(502).json({ error: "upstream_error" });
      }
    });

    // 资源路由
    this.app.post("/resources/read", async (req, res) => {
      const { uri } = req.body;
      const servers = this.registry.findServers("resources");

      if (servers.length === 0) {
        return res.status(503).json({ error: "no_healthy_servers" });
      }

      const cacheKey = `resource:${uri}`;
      const cached = await this.cache.get(cacheKey);
      if (cached) {
        return res.json(cached);
      }

      const server = this.selectServer(servers);
      const result = await this.proxyRequest(server, {
        method: "resources/read",
        params: { uri },
      });

      // 资源可以缓存更长时间
      await this.cache.set(cacheKey, result, 300_000);

      res.json(result);
    });
  }

  // 负载均衡策略
  private selectServer(
    servers: MCPServerRegistration[]
  ): MCPServerRegistration {
    // 最少连接——选择请求最少的服务器
    return servers.reduce((best, current) =>
      current.health.responseTime < best.health.responseTime
        ? current
        : best
    );
  }

  // 代理请求到 MCP 服务器
  private async proxyRequest(
    server: MCPServerRegistration,
    request: { method: string; params: unknown }
  ): Promise<unknown> {
    const response = await fetch(server.endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`服务器 ${server.id} 响应错误：${response.status}`);
    }

    return response.json();
  }

  // 注册新服务器
  registerServer(registration: MCPServerRegistration) {
    this.registry.register(registration);
    console.error(`服务器已注册：${registration.name} (${registration.id})`);
  }

  start(port: number) {
    this.app.listen(port, () => {
      console.log(`MCP 网关运行在端口 ${port}`);
    });
  }
}
```

## 缓存层

```typescript
// cache.ts
interface CacheEntry {
  data: unknown;
  expiresAt: number;
}

class MCPCache {
  private cache: Map<string, CacheEntry> = new Map();
  private maxSize: number;

  constructor(maxSize = 1000) {
    this.maxSize = maxSize;
  }

  async get(key: string): Promise<unknown | null> {
    const entry = this.cache.get(key);

    if (!entry) {
      return null;
    }

    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  async set(
    key: string,
    data: unknown,
    ttlMs: number
  ): Promise<void> {
    // 如果达到最大缓存大小则逐出最旧条目
    if (this.cache.size >= this.maxSize) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey) {
        this.cache.delete(oldestKey);
      }
    }

    this.cache.set(key, {
      data,
      expiresAt: Date.now() + ttlMs,
    });
  }

  async invalidate(pattern: string): Promise<void> {
    const regex = new RegExp(pattern);
    for (const key of this.cache.keys()) {
      if (regex.test(key)) {
        this.cache.delete(key);
      }
    }
  }

  getSize(): number {
    return this.cache.size;
  }
}
```

## 速率限制器

```typescript
// rate-limiter.ts
interface RateLimiterConfig {
  windowMs: number;
  maxRequests: number;
}

interface ClientRate {
  count: number;
  windowStart: number;
}

class RateLimiter {
  private clients: Map<string, ClientRate> = new Map();
  private config: RateLimiterConfig;

  constructor(config: RateLimiterConfig) {
    this.config = config;
  }

  async checkLimit(clientId: string): Promise<boolean> {
    const now = Date.now();
    const clientRate = this.clients.get(clientId);

    if (!clientRate || now - clientRate.windowStart > this.config.windowMs) {
      // 新窗口
      this.clients.set(clientId, {
        count: 1,
        windowStart: now,
      });
      return true;
    }

    if (clientRate.count >= this.config.maxRequests) {
      return false;
    }

    clientRate.count++;
    return true;
  }

  getRetryAfter(clientId: string): number {
    const clientRate = this.clients.get(clientId);
    if (!clientRate) return 0;

    const elapsed = Date.now() - clientRate.windowStart;
    return Math.max(0, this.config.windowMs - elapsed);
  }
}
```

## 自动化服务器注册

```typescript
// auto-registration.ts
async function registerWithGateway(
  gatewayUrl: string,
  serverInfo: {
    name: string;
    version: string;
    endpoint: string;
    transport: MCPServerRegistration["transport"];
  }
) {
  const response = await fetch(`${gatewayUrl}/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: serverInfo.name,
      version: serverInfo.version,
      endpoint: serverInfo.endpoint,
      transport: serverInfo.transport,
      capabilities: {
        tools: true,
        resources: false,
        prompts: false,
      },
      metadata: {
        tools: 5,
        resources: 0,
        prompts: 0,
      },
    }),
  });

  if (!response.ok) {
    throw new Error(`注册失败：${response.statusText}`);
  }

  const registration = await response.json();
  console.log(`已注册为服务器：${registration.id}`);

  // 定期心跳
  setInterval(async () => {
    await fetch(`${gatewayUrl}/health/${registration.id}`, {
      method: "POST",
    });
  }, 15_000);
}
```

## 最佳实践

1. **健康检查**：网关应主动监控服务器健康状况并路由到健康实例。
2. **优雅降级**：当首选服务器不可用时实现故障转移。
3. **缓存策略**：为静态资源使用较长的 TTL，为动态工具调用使用较短的 TTL。
4. **速率限制**：在网关层实现全局和每客户端速率限制。
5. **服务发现**：使用自动化注册，使服务器无需手动配置即可接入。

## 练习

1. **基本 MCP 网关**：实现一个路由到两个 MCP 服务器的简单网关。

2. **网关健康仪表盘**：构建一个 Web UI，显示所有已注册 MCP 服务器的状态和健康情况。

3. **缓存工具结果**：向你的网关添加智能缓存，仅在工具参数变化时缓存幂等工具调用。

4. **加权负载均衡**：实现考虑服务器容量和当前负载的权重负载均衡策略。

5. **多区域网关**：设计一个跨地理区域分发 MCP 请求的网关架构。

## 术语表

- **网关**：充当 MCP 服务器前门的中间层，提供路由和负载均衡。
- **注册中心**：可用 MCP 服务器及其能力的中央目录。
- **健康检查**：定期检查以验证 MCP 服务器是否正常运行。
- **负载均衡**：跨多个服务器分发请求以优化资源使用。
- **端点**：MCP 服务器监听传入连接的网络地址。

## 延伸阅读

- API 网关模式（微服务架构）
- 服务发现模式
- 负载均衡算法
- 分布式系统缓存策略
