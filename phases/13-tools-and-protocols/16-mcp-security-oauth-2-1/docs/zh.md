# MCP 安全：OAuth 2.1

> 使用 OAuth 2.1 保护 MCP 通信，涵盖授权码流程、PKCE、令牌管理和安全部署。

**类型：** 构建（Build）
**语言：** TypeScript
**前置要求：** MCP 基础，HTTP 认证概念
**时间：** ~50 分钟

## 学习目标

- 理解 OAuth 2.1 及其与 OAuth 2.0 的区别
- 使用 OAuth 2.1 保护 MCP 服务器通信
- 实现 PKCE（Proof Key for Code Exchange）
- 管理访问令牌和刷新令牌生命周期
- 将 MCP 认证集成到客户端应用中

## 什么是 OAuth 2.1？

OAuth 2.1 是 OAuth 2.0 授权框架的演进版本，整合了多年来的最佳实践。它简化和加强了核心协议，同时移除了不安全的模式。

```
OAuth 2.0 → OAuth 2.1 关键变化：

❌ 隐式流程（已移除）
❌ 资源拥有者密码凭证（已移除）
❌ 无认证的授权码（已移除）
✅ 始终使用 PKCE 授权码流程（必需）
✅ 刷新令牌轮换（推荐）
✅ 发送者受限令牌（推荐）
```

### 核心原则

1. **授权码 + PKCE 始终**：所有公共客户端都必须使用 PKCE
2. **无隐式流程**：隐式流程已被移除，因为 PKCE 覆盖了所有情况
3. **令牌安全**：令牌应通过加密通道传输，且永不暴露在 URL 中
4. **范围限制**：令牌应具有允许操作的最小范围

## MCP 服务器 OAuth 认证

```typescript
// oauth-server.ts
import express from "express";
import crypto from "crypto";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";

interface ClientRegistration {
  clientId: string;
  clientName: string;
  redirectUris: string[];
  scopes: string[];
}

interface AuthorizationCode {
  code: string;
  clientId: string;
  codeChallenge: string;
  codeChallengeMethod: "S256" | "plain";
  redirectUri: string;
  scope: string[];
  expiresAt: Date;
}

interface AccessToken {
  token: string;
  clientId: string;
  scope: string[];
  expiresAt: Date;
  refreshToken?: string;
}

class MCPOAuthServer {
  private clients: Map<string, ClientRegistration> = new Map();
  private authCodes: Map<string, AuthorizationCode> = new Map();
  private accessTokens: Map<string, AccessToken> = new Map();
  private refreshTokens: Map<string, string> = new Map(); // refresh -> access token ID
  private mcpServer: Server;
  private app: express.Application;

  constructor(mcpServer: Server) {
    this.mcpServer = mcpServer;
    this.app = express();
    this.setupRoutes();
  }

  // 注册 MCP 客户端
  registerClient(client: ClientRegistration) {
    this.clients.set(client.clientId, client);
  }

  private setupRoutes() {
    this.app.use(express.json());

    // OAuth 2.1 授权端点
    this.app.get("/authorize", async (req, res) => {
      const {
        response_type,
        client_id,
        redirect_uri,
        code_challenge,
        code_challenge_method,
        scope,
        state,
      } = req.query;

      // 验证请求
      if (response_type !== "code") {
        return this.sendError(res, "unsupported_response_type", state as string);
      }

      const client = this.clients.get(client_id as string);
      if (!client || !client.redirectUris.includes(redirect_uri as string)) {
        return this.sendError(res, "unauthorized_client", state as string);
      }

      // 生成授权码
      const code = this.generateCode();
      const authCode: AuthorizationCode = {
        code,
        clientId: client_id as string,
        codeChallenge: code_challenge as string,
        codeChallengeMethod: (code_challenge_method as "S256" | "plain") || "S256",
        redirectUri: redirect_uri as string,
        scope: (scope as string)?.split(" ") || client.scopes,
        expiresAt: new Date(Date.now() + 10 * 60 * 1000), // 10 分钟
      };

      this.authCodes.set(code, authCode);

      // 重定向回客户端
      const redirectUrl = new URL(redirect_uri as string);
      redirectUrl.searchParams.set("code", code);
      if (state) {
        redirectUrl.searchParams.set("state", state as string);
      }

      res.redirect(redirectUrl.toString());
    });

    // 令牌端点
    this.app.post("/token", async (req, res) => {
      const {
        grant_type,
        code,
        code_verifier,
        redirect_uri,
        client_id,
        refresh_token,
      } = req.body;

      if (grant_type === "authorization_code") {
        await this.handleAuthorizationCodeGrant(
          res,
          code,
          code_verifier,
          redirect_uri,
          client_id
        );
      } else if (grant_type === "refresh_token") {
        await this.handleRefreshTokenGrant(res, refresh_token, client_id);
      } else {
        this.sendError(res, "unsupported_grant_type");
      }
    });

    // MCP 端点（令牌保护）
    this.app.post("/mcp", async (req, res) => {
      const authHeader = req.headers.authorization;
      if (!authHeader || !authHeader.startsWith("Bearer ")) {
        return res.status(401).json({ error: "invalid_token" });
      }

      const token = authHeader.slice(7);
      const accessToken = this.accessTokens.get(token);

      if (!accessToken || accessToken.expiresAt < new Date()) {
        return res.status(401).json({ error: "token_expired" });
      }

      // 转发请求到 MCP 服务器
      await this.handleMCPRequest(req, res, accessToken);
    });
  }

  private async handleAuthorizationCodeGrant(
    res: express.Response,
    code: string,
    codeVerifier: string,
    redirectUri: string,
    clientId: string
  ) {
    // 验证授权码
    const authCode = this.authCodes.get(code);
    if (!authCode || authCode.expiresAt < new Date()) {
      return this.sendError(res, "invalid_grant", undefined, 400);
    }

    // 验证 PKCE
    if (!this.verifyPKCE(codeVerifier, authCode)) {
      return this.sendError(res, "invalid_grant", undefined, 400);
    }

    // 验证 redirect_uri
    if (authCode.redirectUri !== redirectUri) {
      return this.sendError(res, "invalid_grant", undefined, 400);
    }

    // 生成令牌
    const accessToken = this.generateAccessToken(authCode);
    const refreshToken = crypto.randomBytes(32).toString("hex");

    this.accessTokens.set(accessToken.token, accessToken);
    this.refreshTokens.set(refreshToken, accessToken.token);

    // 清理
    this.authCodes.delete(code);

    res.json({
      access_token: accessToken.token,
      token_type: "Bearer",
      expires_in: 3600,
      refresh_token: refreshToken,
      scope: accessToken.scope.join(" "),
    });
  }

  private async handleRefreshTokenGrant(
    res: express.Response,
    refreshToken: string,
    clientId: string
  ) {
    const accessTokenId = this.refreshTokens.get(refreshToken);
    if (!accessTokenId) {
      return this.sendError(res, "invalid_grant", undefined, 400);
    }

    const oldToken = this.accessTokens.get(accessTokenId);
    if (!oldToken || oldToken.clientId !== clientId) {
      return this.sendError(res, "invalid_grant", undefined, 400);
    }

    // OAuth 2.1 刷新令牌轮换
    const newAccessToken = this.generateAccessToken({
      clientId: oldToken.clientId,
      scope: oldToken.scope,
    } as AuthorizationCode);

    const newRefreshToken = crypto.randomBytes(32).toString("hex");

    this.accessTokens.set(newAccessToken.token, newAccessToken);
    this.refreshTokens.set(newRefreshToken, newAccessToken.token);

    // 轮换：移除旧令牌
    this.accessTokens.delete(accessTokenId);
    this.refreshTokens.delete(refreshToken);

    res.json({
      access_token: newAccessToken.token,
      token_type: "Bearer",
      expires_in: 3600,
      refresh_token: newRefreshToken,
      scope: newAccessToken.scope.join(" "),
    });
  }

  private verifyPKCE(
    codeVerifier: string,
    authCode: AuthorizationCode
  ): boolean {
    if (authCode.codeChallengeMethod === "S256") {
      const hash = crypto
        .createHash("sha256")
        .update(codeVerifier)
        .digest("base64url");
      return hash === authCode.codeChallenge;
    } else {
      return codeVerifier === authCode.codeChallenge;
    }
  }

  private generateCode(): string {
    return crypto.randomBytes(16).toString("hex");
  }

  private generateAccessToken(authCode: {
    clientId: string;
    scope: string[];
  }): AccessToken {
    return {
      token: crypto.randomBytes(24).toString("hex"),
      clientId: authCode.clientId,
      scope: authCode.scope,
      expiresAt: new Date(Date.now() + 3600 * 1000), // 1 小时
    };
  }

  private sendError(
    res: express.Response,
    error: string,
    state?: string,
    status = 400
  ) {
    const body: Record<string, string> = { error };
    if (state) body.state = state;
    res.status(status).json(body);
  }

  private async handleMCPRequest(
    req: express.Request,
    res: express.Response,
    token: AccessToken
  ) {
    // 根据令牌范围验证 MCP 请求
    // 将请求转发到 MCP 服务器
    res.json({ success: true, scope: token.scope });
  }

  start(port: number) {
    this.app.listen(port, () => {
      console.log(`MCP OAuth 服务器运行在端口 ${port}`);
    });
  }
}
```

## MCP 客户端 OAuth 集成

```typescript
// oauth-client.ts
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { createHash, randomBytes } from "crypto";

class MCPOAuthClient {
  private client: Client;
  private config: {
    clientId: string;
    authorizationEndpoint: string;
    tokenEndpoint: string;
    redirectUri: string;
    scopes: string[];
  };
  private tokens: {
    accessToken?: string;
    refreshToken?: string;
    expiresAt?: Date;
  } = {};

  constructor(config: {
    clientId: string;
    authorizationEndpoint: string;
    tokenEndpoint: string;
    redirectUri: string;
    scopes?: string[];
  }) {
    this.client = new Client(
      { name: "mcp-oauth-client", version: "1.0.0" },
      { capabilities: {} }
    );

    this.config = {
      ...config,
      scopes: config.scopes || ["mcp"],
    };
  }

  // 启动 OAuth 2.1 授权码 + PKCE 流程
  async authorize(): Promise<void> {
    // 1. 生成 PKCE 挑战
    const codeVerifier = this.generateCodeVerifier();
    const codeChallenge = this.generateCodeChallenge(codeVerifier);

    // 2. 构建授权 URL
    const authUrl = new URL(this.config.authorizationEndpoint);
    authUrl.searchParams.set("response_type", "code");
    authUrl.searchParams.set("client_id", this.config.clientId);
    authUrl.searchParams.set("redirect_uri", this.config.redirectUri);
    authUrl.searchParams.set("code_challenge", codeChallenge);
    authUrl.searchParams.set("code_challenge_method", "S256");
    authUrl.searchParams.set("scope", this.config.scopes.join(" "));
    authUrl.searchParams.set("state", this.generateState());

    // 3. 重定向用户代理进行授权
    console.log("重定向到授权服务器...");
    console.log(`授权 URL: ${authUrl.toString()}`);

    // 在 Web 应用中，这会是浏览器重定向
    // 对于 CLI，打印 URL 并等待回调
    const authorizationCode = await this.waitForCallback();

    // 4. 用授权码交换令牌
    await this.exchangeCodeForToken(authorizationCode, codeVerifier);
  }

  private generateCodeVerifier(): string {
    return randomBytes(32)
      .toString("base64")
      .replace(/[+/=]/g, "")
      .slice(0, 128);
  }

  private generateCodeChallenge(verifier: string): string {
    return createHash("sha256")
      .update(verifier)
      .digest("base64")
      .replace(/[+/=]/g, "");
  }

  private generateState(): string {
    return randomBytes(16).toString("hex");
  }

  private async waitForCallback(): Promise<string> {
    // 在 CLI 中，这可以是一个重定向服务器
    // 在 Web 中，是 URL 重定向处理程序
    // 这里为了简单起见模拟它
    return new Promise((resolve) => {
      console.log("输入授权码：");
      process.stdin.once("data", (data) => {
        resolve(data.toString().trim());
      });
    });
  }

  private async exchangeCodeForToken(
    code: string,
    codeVerifier: string
  ): Promise<void> {
    const response = await fetch(this.config.tokenEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        grant_type: "authorization_code",
        code,
        code_verifier: codeVerifier,
        redirect_uri: this.config.redirectUri,
        client_id: this.config.clientId,
      }),
    });

    if (!response.ok) {
      throw new Error(`令牌请求失败：${response.statusText}`);
    }

    const tokenResponse = await response.json();
    this.setTokens(tokenResponse);
  }

  private setTokens(tokenResponse: {
    access_token: string;
    expires_in: number;
    refresh_token?: string;
  }) {
    this.tokens = {
      accessToken: tokenResponse.access_token,
      refreshToken: tokenResponse.refresh_token,
      expiresAt: new Date(Date.now() + tokenResponse.expires_in * 1000),
    };
  }

  // OAuth 2.1 刷新令牌轮换
  async refreshAccessToken(): Promise<void> {
    if (!this.tokens.refreshToken) {
      throw new Error("无刷新令牌可用");
    }

    const response = await fetch(this.config.tokenEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        grant_type: "refresh_token",
        refresh_token: this.tokens.refreshToken,
        client_id: this.config.clientId,
      }),
    });

    if (!response.ok) {
      // 刷新令牌轮换失败——需要重新授权
      this.tokens = {};
      await this.authorize();
      return;
    }

    const tokenResponse = await response.json();
    this.setTokens(tokenResponse);
  }

  // MCP 客户端认证的连接
  async connectToMCPServer(mcpUrl: string): Promise<void> {
    // 确保我们有有效的令牌
    if (
      !this.tokens.accessToken ||
      (this.tokens.expiresAt && this.tokens.expiresAt < new Date())
    ) {
      if (this.tokens.refreshToken) {
        await this.refreshAccessToken();
      } else {
        throw new Error("需要授权。先调用 authorize()。");
      }
    }

    // 使用承载令牌连接
    const { SSEClientTransport } = await import(
      "@modelcontextprotocol/sdk/client/sse.js"
    );

    const transport = new SSEClientTransport(new URL(mcpUrl));

    // 添加认证头部
    // 注意：实际的 MCP SDK 可能以不同方式处理此问题
    await this.client.connect(transport, {
      authorization: `Bearer ${this.tokens.accessToken}`,
    });
  }

  getAccessToken(): string | undefined {
    return this.tokens.accessToken;
  }

  isAuthorized(): boolean {
    return !!this.tokens.accessToken;
  }
}
```

## 范围管理

```typescript
// scope-manager.ts
interface Scope {
  name: string;
  description: string;
  resources: string[];
  tools: string[];
}

class MCPScopeManager {
  private scopes: Map<string, Scope> = new Map();

  constructor() {
    this.registerDefaultScopes();
  }

  private registerDefaultScopes() {
    this.registerScope({
      name: "mcp:read",
      description: "读取 MCP 资源和提示词",
      resources: ["*"],
      tools: ["list"],
    });

    this.registerScope({
      name: "mcp:write",
      description: "写入 MCP 资源和执行工具（隐式包含读取）",
      resources: ["*"],
      tools: ["*"],
    });

    this.registerScope({
      name: "mcp:admin",
      description: "完全管理访问",
      resources: ["*"],
      tools: ["*"],
      additionalPermissions: ["manage_clients", "manage_scopes"],
    });
  }

  registerScope(scope: Scope) {
    this.scopes.set(scope.name, scope);
  }

  hasPermission(
    requiredScope: string,
    tokenScopes: string[]
  ): boolean {
    // 检查令牌范围是否包含所需范围
    return tokenScopes.some((tokenScope) => {
      const scope = this.scopes.get(tokenScope);
      if (!scope) return false;

      // 通配符检查
      if (tokenScope === "mcp:admin") return true;
      if (tokenScope === "mcp:write" && requiredScope === "mcp:read") {
        return true; // write 隐式包含 read
      }

      return tokenScope === requiredScope;
    });
  }
}
```

## 最佳实践

1. **始终使用 PKCE**：所有客户端都需要 PKCE，包括所谓的"机密客户端"。
2. **刷新令牌轮换**：每次使用刷新令牌时都要轮换，旧令牌立即失效。
3. **短生命周期访问令牌**：访问令牌应在 1 小时内过期。
4. **HTTPS 必须**：所有 OAuth 端点必须通过 HTTPS 提供。
5. **范围最小化**：令牌的范围应尽可能小（最小权限原则）。

## 练习

1. **OAuth 授权服务器**：为你的 MCP 服务器实现完整的 OAuth 2.1 授权服务器。

2. **PKCE CLI 流程**：创建一个使用 PKCE 授权码流程的 CLI MCP 客户端。

3. **令牌管理 UI**：构建一个 Web 界面，用于管理 MCP 令牌、查看范围和撤销访问。

4. **范围定制**：为你的 MCP 服务器的特定工具和资源实现细粒度的范围。

5. **安全审计**：审计现有的 MCP 实现，检查 OAuth 2.1 合规性和任何安全漏洞。

## 术语表

- **OAuth 2.1**：结合 PKCE 和不安全流程移除的演进版 OAuth 2.0。
- **PKCE**：Proof Key for Code Exchange——一种在不使用客户端密钥的情况下保障授权码流程安全的技术。
- **授权码**：用户在授权后收到的临时码，客户端用它交换访问令牌。
- **访问令牌**：客户端用于验证 API 请求的短期凭证。
- **刷新令牌**：用于在访问令牌过期后获取新令牌的长期凭证。

## 延伸阅读

- OAuth 2.1 规范（IETF 草案）
- PKCE 规范（RFC 7636）
- OAuth 2.0 安全最佳实践（RFC 9700）
- MCP 认证规范
