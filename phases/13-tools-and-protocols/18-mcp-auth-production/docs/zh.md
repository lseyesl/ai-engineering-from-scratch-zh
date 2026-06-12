# MCP 生产级认证 — 客户端注册、JWKS 刷新、受众绑定令牌

> 第 16 课在内存中搭建了 OAuth 2.1 状态机。到 2026 年，你交付给真实组织的每一个 MCP 服务器都位于生产级认证之后：可扩展至无限客户端群体的客户端注册（优先使用客户端 ID 元数据文档，动态客户端注册作为向后兼容的回退方案）、授权服务器元数据发现（RFC 8414 *或* OpenID Connect Discovery）、不会在凌晨 3 点令牌验证中断的 JWKS 缓存刷新，以及拒绝跨资源重放的受众绑定令牌。本课用三个角色建模完整的安全面——授权服务器、资源服务器（MCP 服务器）和客户端——让你可以追踪从发现到验证工具调用的每一跳。
>
> **规范说明（2025-11-25）：** 2025 年 11 月的 MCP 授权规范将动态客户端注册从 `SHOULD` 降级为 `MAY`，并将**客户端 ID 元数据文档（CIMD）**设为推荐的默认注册机制。本课按规范的优先顺序教授两者，代码演练保留 DCR 是因为它可以在单进程中完全自包含。

**类型：** 构建（Build）
**语言：** Python（标准库）
**前置要求：** 第 13 阶段 · 第 16 课（OAuth 2.1 状态机），第 13 阶段 · 第 17 课（网关）
**时间：** ~90 分钟

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器并验证其契约。
- 实现 RFC 7591 动态客户端注册，使 MCP 客户端无需管理员干预即可注册。
- 按计划缓存和刷新 JWKS 密钥，使签名验证在密钥轮换后仍能正常工作。
- 使用 RFC 8707 资源指示器将令牌绑定到单个 MCP 资源，拒绝混淆代理重用。
- 清晰分离三个角色——授权服务器、资源服务器、客户端——使每个角色只执行属于自己的检查。
- 阅读 IdP 能力矩阵，当 IdP 无法满足 MCP 的认证配置文件时拒绝部署。

## 问题

第 16 课的模拟器在内存中运行 OAuth 2.1。生产环境有三个仅靠内存模拟器无法看到的运维缺口。

第一个缺口是注册。一个真实的组织运行数百个 MCP 服务器和数千个 MCP 客户端。运维人员不会手工注册每一个 Cursor 用户作为 OAuth 客户端。2025-11-25 规范为客户端提供了解决此问题的优先顺序：如果你有预注册的 `client_id` 则使用它，否则使用**客户端 ID 元数据文档**（客户端用自己控制的 HTTPS URL 标识自身，授权服务器*拉取*元数据），否则回退到 **RFC 7591 动态客户端注册**（客户端*推送*一个 `POST /register` 并当场获得 `client_id`），否则提示用户。CIMD 是推荐的默认方案，因为它完全消除了逐服务器注册，同时保持以 DNS 为根的信任模型；DCR 保留用于向后兼容。两者都从授权服务器的元数据中发现其入口点：CIMD 对应 `client_id_metadata_document_supported`，DCR 对应 `registration_endpoint`。

第二个缺口是密钥轮换。JWT 验证依赖授权服务器的签名密钥，以 JSON Web Key Set（JWKS）形式发布。授权服务器按计划轮换这些密钥（通常每小时，在事件响应期间有时更快）。只在启动时获取一次 JWKS 的 MCP 服务器在轮换窗口之前验证正常——然后每个请求都会失败，直到重启。生产环境将 JWKS 配置为带刷新任务的缓存值，在旧密钥过期之前覆盖缓存，加上缓存未命中时的回退获取，以处理由比缓存更新的密钥签名的令牌到达的情况。

第三个缺口是受众绑定。第 16 课引入了 RFC 8707 资源指示器。在生产环境中，该指示器成为每个请求上的硬性声明检查。MCP 服务器将 `token.aud` 与自身的规范资源 URL 进行比较，不匹配时以 HTTP 401 拒绝。这是防御上游 MCP 服务器（或持有为某个服务器签发的令牌的恶意客户端）在同一信任网格中对另一个服务器重放该令牌的唯一手段。

本课将每个缺口映射到安全面的具体部分。元数据文档是一个 HTTP 端点。JWKS 缓存刷新是一个定时任务加键值缓存。JWT 验证是资源服务器在分派任何工具之前运行的例程。保持三个角色分离，每个角色只执行属于自己的检查：授权服务器签发和轮换密钥，资源服务器缓存和验证，客户端发现和注册。

## 核心概念

### RFC 8414 — OAuth 授权服务器元数据

位于 `/.well-known/oauth-authorization-server` 的文档描述了客户端需要的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

给定 MCP 资源 URL 的客户端链接发现：RFC 9728 的 `oauth-protected-resource`（资源服务器的文档）命名了签发者，然后 `oauth-authorization-server`（本 RFC）命名了每个端点。客户端从不硬编码授权 URL。

在信任 IdP 用于 MCP 之前需要验证的契约：

- `code_challenge_methods_supported` 包含 `S256`（RFC 7636 的 PKCE）。规范明确指出：如果此字段**缺失**，授权服务器不支持 PKCE，客户端**必须**拒绝继续。
- `grant_types_supported` 包含 `authorization_code` 并拒绝 `password` 和 `implicit`。
- 至少公布一个注册路径：`client_id_metadata_document_supported: true`（CIMD，优先）**或** `registration_endpoint`（RFC 7591 DCR，回退）。任一都满足契约；你不再硬性要求 DCR。
- `response_types_supported` 对于 OAuth 2.1 恰好是 `["code"]`。

如果 `S256` 缺失，MCP 服务器拒绝针对此 IdP 部署——PKCE 没有降级模式。如果*两个*注册路径都未公布且你没有预注册的 `client_id`，你也无法注册；部署清单有误，不是代码的问题。

### RFC 9728（回顾）— 受保护资源元数据

第 16 课涵盖了 RFC 9728。生产环境的增量：此文档是客户端查找*此* MCP 服务器信任的授权服务器的唯一位置。单个 MCP 服务器可以接受来自多个 IdP 的令牌（一个用于员工，一个用于合作伙伴）。RFC 9728 声明该集合；RFC 8414 记录每个 IdP 支持什么。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### 客户端 ID 元数据文档（推荐的默认方案）

CIMD 将注册从*推送*反转为*拉取*。客户端不再请求授权服务器铸造 `client_id`，而是使用自己控制的 HTTPS URL **作为**其 `client_id`。该 URL 解析为一个 JSON 元数据文档；授权服务器在 OAuth 流程中按需获取它。信任根植于 DNS：如果服务器运维人员信任 `app.example.com`，就信任从 `https://app.example.com/client.json` 提供服务的客户端。无需注册往返，没有 `client_id` 命名空间会耗尽，无需保持每服务器状态同步。

客户端托管的元数据文档：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

文档中的 `client_id` 值**必须**等于其提供服务的 URL（授权服务器验证此点；不匹配则被拒绝）。授权服务器通过其 RFC 8414 元数据中的 `client_id_metadata_document_supported: true` 公告支持。

规范直言不讳的两个安全事实：

- **SSRF。** 授权服务器获取攻击者提供的 URL。它必须防御服务器端请求伪造（不获取内部/管理端点）。
- **localhost 冒充。** 仅靠 CIMD 无法阻止本地攻击者声称合法客户端的元数据 URL 并绑定任何 `localhost` 重定向。授权服务器**必须**在同意页面清晰显示重定向 URI 主机名，并**应该**对仅 `localhost` 的重定向发出警告。

因为 CIMD 不需要服务器端状态，所以没有 DCR 所需的注册器需要搭建。客户端侧是只读的：从静态 HTTPS 端点提供元数据文档，让授权服务器拉取它。

### RFC 7591 — 动态客户端注册（回退 / 向后兼容）

DCR 现在是 `MAY`，保留用于与 2025-11-25 之前部署和不支持 CIMD 的 IdP 的向后兼容。没有它（也没有 CIMD 或预注册），每个 MCP 客户端（Cursor、Claude Desktop、自定义代理）都需要与 IdP 管理员进行带外交换。使用 DCR，客户端发送：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器响应 `client_id` 和用于后续更新的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` 是运行在用户设备上的 MCP 客户端的正确默认值。它们只获得 `client_id`——没有可泄露的 `client_secret`。PKCE 提供公共客户端所需的持有者证明。

三个生产陷阱：

- 注册端点必须按源 IP 限速。否则，恶意行为者可以脚本化数百万虚假注册并耗尽 `client_id` 命名空间。在注册器处理请求之前运行限速检查。
- 某些企业 IdP 要求 `software_statement`（一个为客户端担保的签名 JWT）。本课的模拟跳过了它；生产环境需要接入验证步骤，拒绝来自非 localhost 重定向 URI 的未签名注册。
- `registration_access_token` 必须以哈希形式存储，而非明文。此令牌被盗意味着攻击者可以重写客户端的重定向 URI。

### RFC 8707（回顾）— 资源指示器

第 16 课确立了其形式。生产规则：每个令牌请求包含 `resource=<canonical-mcp-url>`，MCP 服务器在每次调用时验证 `token.aud` 匹配自身的资源 URL。规范 URI 是服务器的*最具体*标识符：使用小写方案和主机，无片段，按惯例无尾部斜杠。路径组件**不**按规则剥离——当需要区分单个 MCP 服务器时，规范保留它。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443` 和 `https://mcp.example.com/server/mcp` 都是有效的规范 URI。每个服务器选一个，并将 `aud` 精确绑定到它。（本课的模拟为简洁起见使用裸主机受众如 `https://notes.example.com`；在同一源下托管多个 MCP 服务器的部署通过路径区分它们。）

### RFC 7636（回顾）— PKCE

PKCE 在 OAuth 2.1 中是强制性的。本课的授权码流程始终携带 `code_challenge` 和 `code_verifier`。服务器拒绝任何没有验证器或验证器不哈希到存储的质询的令牌请求。

### MCP 规范 2025-11-25 认证配置文件

MCP 规范（2025-11-25）对 MCP 服务器的授权层必须做什么有精确要求：

- 实现 RFC 9728 受保护资源元数据，并通过 401 上的 `WWW-Authenticate: Bearer resource_metadata="..."` 头**或**知名 URI `/.well-known/oauth-protected-resource` 提供其位置（SEP-985 使头部可选，提供知名回退）。元数据的 `authorization_servers` 字段**必须**命名至少一个服务器。
- 在**每个**请求上仅通过 `Authorization: Bearer ...` 接受令牌——从不在查询字符串中，从不仅在会话开始时验证。
- 每个请求验证 `aud`、`iss`、`exp` 和所需范围。服务器**必须**验证令牌是专门为其签发的（受众）；缺失或不匹配的 `aud` 被拒绝，从不视为通配符。
- 在 401/403 时，返回 `WWW-Authenticate: Bearer`，携带 `error=...`、`resource_metadata="<PRM-URL>"` 参数（元数据文档的 URL，*不是*裸资源）和 `insufficient_scope`（403）时的 `scope="..."`。注意：参数是 `resource_metadata`，一个发现指针——质询中没有 `resource` 参数。
- 授权服务器发现接受 **RFC 8414 OAuth 元数据** **或** OpenID Connect Discovery 1.0；客户端必须按优先顺序尝试两个知名后缀。
- 客户端（而非服务器）防御**混淆攻击**：在重定向前记录预期的 `issuer`，并在兑换授权码之前验证 `iss` 授权响应参数（RFC 9207）。仅靠 PKCE 不能阻止混淆，因为客户端将其 `code_verifier` 交给它被引导到的任何令牌端点。

OAuth 2.1 草案是基底；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD 是安全面；MCP 规范是配置文件。

### IdP 能力矩阵

并非每个 IdP 都支持完整的 MCP 配置文件。下表记录了截至 2025-11-25 规范的事实性能力声明。它是*部署门控*，而非推荐。

CIMD 在 2025-11-25 规范中发布，底层 OAuth 草案仅在 2025 年 10 月被采纳，因此供应商支持仍在陆续到来——将下表中的"CIMD"视为"当前状态，请在你的租户中验证"，而非永久声明。

| IdP 类别 | AS 元数据（8414/OIDC） | CIMD | RFC 7591 DCR | RFC 8707 资源 | RFC 7636 S256 PKCE | 备注 |
|---|---|---|---|---|---|---|
| 自托管（Keycloak） | 是 | 新兴 | 是 | 是（24.x 起） | 是 | 本课 MCP 配置文件的参考 IdP；完整 DCR 路径端到端，CIMD 跟踪新规范。 |
| 企业 SSO（Microsoft Entra ID） | 是 | 新兴 | 是（高级层） | 是 | 是 | DCR 可用性因租户层级而异；部署前在目标租户中验证。 |
| 企业 SSO（Okta） | 是 | 新兴 | 是（Okta CIC / Auth0） | 是 | 是 | DCR 在 Auth0（现为 Okta CIC）上可用；经典 Okta 组织需要管理员预注册。 |
| 社交登录 IdP（通用） | 不定 | 否 | 罕见 | 罕见 | 是 | 大多数社交 IdP 将客户端视为静态合作伙伴；无自助注册。仅用作身份源，在其上层叠你自己的 MCP 感知授权服务器。 |
| 自定义 / 自建 | 取决于实现 | 取决于实现 | 取决于实现 | 取决于实现 | 取决于实现 | 如果你自建，请实现完整配置文件并优先使用 CIMD。跳过 PKCE 或受众绑定会破坏 MCP 认证契约。 |

部署清单的拒绝规则：如果所选 IdP 未在 `code_challenge_methods_supported` 中列出 `S256`，MCP 服务器拒绝启动——PKCE 没有降级模式。注册是较软的门控：你需要*一条*可用路径（预注册的 `client_id`、`client_id_metadata_document_supported: true` 或 `registration_endpoint`）。仅 DCR 的缺失不再是拒绝触发器，因为 CIMD 或预注册可以覆盖它。

### JWKS 刷新模式（在 AS 端轮换，在资源服务器端刷新）

保持两个动词分开，因为混淆它们是真实的生产缺陷：

- **轮换**是*授权服务器*做的事：铸造新签名密钥，在 JWKS 中发布它，稍后退役旧密钥。资源服务器与此无关，也不能执行——它不持有 IdP 的私钥。
- **刷新**是*资源服务器*做的事：重新 `GET` 已发布的 JWKS 到其缓存中。这是资源服务器唯一执行的 JWKS 操作。

生产故障模式是过期缓存。用定时刷新任务加键值缓存解决。资源服务器运行一个任务（cron、定时器，或你的运行时提供的任何机制），按固定间隔获取 `<issuer>/.well-known/jwks.json` 并覆盖 `cache[issuer] = {keys, fetched_at}`。验证器从该缓存读取。令牌的 `kid` 在缓存中缺失时触发**一次**同步刷新作为回退，然后重新检查。这同时处理两种情况：定时刷新，以及由全新密钥签名的令牌在下一次定时刷新之前到达的重叠窗口。

回退**必须是重新获取，绝不能是轮换**。如果你将缓存未命中路径连接到轮换并铸造，两件事会出问题：(1) 铸造新密钥产生的 `kid` *仍然*不匹配令牌，因此查找仍然失败；(2) 用随机 `kid` 值喷洒令牌的攻击者会强制无限系列的密钥创建——自我造成的 DoS。重新获取是幂等的，因此伪造的 `kid` 最多只浪费一次获取。

缓存结构：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时持有两个密钥是稳态。授权服务器通过在退役前一个密钥（`k_2026_03`）之前引入下一个密钥（`k_2026_04`）来轮换，因此在旧密钥下签发的令牌在过期之前仍然有效。缓存持有并集；验证器按 `kid` 选取。

### 验证例程

MCP 服务器在分派任何工具之前运行验证。`code/main.py` 使用的形式：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` 解码 JWT，从 JWKS 缓存解析签名密钥（未命中时刷新一次），验证签名，然后对照允许列表检查 `iss`、对照此服务器的规范资源检查 `aud`、检查 `exp` 和所需范围——在第一次失败时返回 `WWW-Authenticate` 质询。将其保持为资源服务器上的单一例程意味着每个入口点（每个工具调用、每个传输）都经过相同的检查；不存在不先验证就到达工具的路径。

### 受众重放演练（访问令牌权限限制）

服务器 A（`notes.example.com`）和服务器 B（`tasks.example.com`）都注册到同一个授权服务器。服务器 A 被攻破。攻击者获取用户的笔记令牌并将其重放到服务器 B。

服务器 B 的验证器：

1. 解码 JWT，按 `kid` 获取 JWKS，验证签名。
2. 对照其受保护资源元数据的 `authorization_servers` 检查 `iss`。（通过——同一 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败——令牌的 `aud` 是 `https://notes.example.com`。）
4. 返回 401，附带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`。

受众声明是协议层防御此攻击的唯一手段。为了性能跳过它是最常见的生产错误；验证器必须在每个请求上运行，而不仅仅在会话开始时。规范称之为**访问令牌权限限制**：MCP 服务器`必须`拒绝任何未在受众中命名它的令牌。

> **命名说明。** 规范将*混淆代理*一词保留给一个相关但不同的问题：MCP 服务器作为第三方 API 的 OAuth **代理**，使用静态客户端 ID，在未获得每客户端用户同意的情况下转发令牌。受众绑定修复了上面的重放；混淆代理的修复是每客户端同意**加上**绝不将入站令牌传递给上游 API（MCP 服务器`必须`获取自己独立的上游令牌）。

### 混淆攻击（服务器无法提供的客户端侧防御）

客户端在其生命周期中与许多授权服务器通信。恶意 AS 可以尝试让客户端在攻击者的令牌端点兑换诚实 AS 的授权码。受众绑定在这里不起作用——攻击发生在任何令牌存在之前。防御在客户端（RFC 9207）：

1. 重定向前，客户端从已验证的 AS 元数据中记录预期的 `issuer`。
2. 在授权响应上，客户端将返回的 `iss` 参数与记录的签发者进行比较（简单字符串比较，无规范化），然后再将授权码发送到任何地方。
3. 不匹配（或 AS 公告了 `authorization_response_iss_parameter_supported` 但 `iss` 缺失）→ 拒绝，甚至不显示 `error` 字段。

仅靠 PKCE 不能阻止混淆，因为客户端将其 `code_verifier` 交给它被引导到的任何令牌端点。这就是为什么规范在每个请求中与 PKCE 验证器和 `state` 一起记录签发者。

### 故障模式

- **JWKS 过期。** 验证器在 AS 轮换密钥后拒绝有效令牌。修复是上面的定时刷新 + 缓存未命中重新获取模式。绝不要在没有刷新任务的情况下缓存 JWKS。
- **轮换作为回退。** 将缓存未命中路径连接到轮换并铸造而非重新获取是真实缺陷：它永远无法产生缺失的 `kid`，并将攻击者控制的 `kid` 值变成密钥创建 DoS。回退必须是幂等的 `refresh-jwks`。
- **缺失 `aud` 声明。** 某些 IdP 默认在令牌请求中不存在 `resource` 时省略 `aud`。验证器必须拒绝缺失 `aud` 的令牌，而非将缺失视为通配符。
- **因缺失 `iss` 检查导致的混淆。** 不验证 RFC 9207 `iss` 授权响应参数与重定向前记录的签发者的客户端，可能被引导到在攻击者的令牌端点兑换诚实 AS 的授权码。这是客户端侧故障；资源服务器无法弥补。
- **范围升级竞争。** 同一用户的两个并发步骤提升流程可能都成功并产生具有不同范围的两个访问令牌。验证器必须使用请求上呈现的令牌，而非查找"用户当前范围"——那会创建 TOCTOU 窗口。
- **注册令牌被盗。** 泄露的 `registration_access_token` 让攻击者重写重定向 URI。静态存储哈希；要求客户端在每次更新时呈现明文；可疑时轮换。
- **`iss` 未绑定。** 接受任何 `iss` 的验证器允许攻击者搭建自己的授权服务器，为目标受众注册客户端，并签发令牌。受保护资源元数据的 `authorization_servers` 列表是允许列表；执行它。

## 使用它

`code/main.py` 用标准库 Python 和三个角色——`AuthorizationServer`、`ResourceServer` 和 `Client`——演练完整的生产流程。流程如下：

1. 授权服务器在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点并检查其注册选项（CIMD 的 `client_id_metadata_document_supported`，DCR 的 `registration_endpoint`）和 `S256` PKCE 支持。
3. 演练采用 DCR 回退路径：客户端向 `/register` 发送 POST（RFC 7591）并获得 `client_id`。（CIMD 客户端则会呈现自己的 HTTPS `client_id` URL 并跳过此步骤。）
4. MCP 客户端运行 PKCE 保护的授权码流程（RFC 7636），带 `resource` 指示器（RFC 8707）。
5. MCP 客户端使用 `Authorization: Bearer ...` 调用 MCP 服务器上的工具。
6. MCP 服务器运行 `validate`，从 JWKS 缓存解析签名密钥。
7. IdP 轮换密钥；定时刷新将 JWKS 重新拉取到缓存中。
8. 下一次调用使用刷新后的密钥验证而无需重启，且前一个令牌在重叠窗口期间仍然有效。
9. 针对不同 MCP 资源的受众重放尝试获得 401，附带 `audience mismatch` 和 `resource_metadata` 指针。

此处的 JWT 使用 HS256 和共享密钥（因此本课仅用标准库即可运行）。生产环境使用 RS256 或 EdDSA 以及上述 JWKS 模式；验证逻辑除此之外完全相同。因为 IdP 和资源服务器位于同一进程中，`refresh_jwks` 直接读取授权服务器的密钥列表；在线上它是到 `jwks_uri` 的 HTTP `GET`。

## 交付它

本课产出 `outputs/skill-mcp-auth.md`。给定 MCP 服务器配置和 IdP 能力集，该技能输出需要搭建的安全面——受保护资源元数据、要使用的注册路径（CIMD、预注册或 DCR 回退）、JWKS 刷新计划、范围映射，以及当 IdP 不支持完整 RFC 配置文件时应用的拒绝规则。

## 练习

1. 运行 `code/main.py`。追踪流程。注意 IdP 在步骤 6 中如何轮换密钥，定时 `refresh_jwks` 如何重新拉取已发布的密钥集，以及旧令牌（重叠窗口）和新令牌如何在无需重启的情况下验证通过。

2. 向受保护资源元数据的 `authorization_servers` 列表添加新 IdP。签发由新 IdP 签名的令牌并确认验证器接受它。签发由未列出的 IdP 签名的令牌并确认验证器以 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"` 拒绝。

3. 向 `register_client` 添加限速检查，在注册器接受请求之前运行。使用按源 IP 的令牌桶，保存在以 IP 为键的小型字典中。

4. 阅读 RFC 7591 并识别本课 `/register` 处理程序未验证的两个字段。添加验证。（提示：`software_statement` 和 `redirect_uris` URI 方案。）

5. 添加客户端 ID 元数据文档路径。提供 `client.json`，其 `client_id` 等于自身的 URL，并让授权服务器获取并验证它（如果 `client_id` ≠ URL 则拒绝）。确认 CIMD 客户端无需 `register_client` 调用即可注册。

6. 证明 DoS 修复。向验证器发送具有随机 `kid` 的令牌，确认 `refresh_jwks` 最多运行一次且授权服务器的密钥计数不增长。然后故意将回退重新连接为轮换并铸造，观察密钥计数随每个伪造令牌攀升——之后恢复为重新获取。

7. 实现混淆部分的客户端侧 RFC 9207 `iss` 检查：在授权请求之前记录预期签发者，然后拒绝 `iss` 不匹配的授权响应。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| ASM | "OAuth 元数据文档" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "客户端元数据 URL" | 客户端 ID 元数据文档——用作 `client_id` 的 HTTPS URL；AS 拉取 JSON。自 2025-11-25 起为推荐默认方案 |
| DCR | "自助客户端注册" | RFC 7591 `POST /register` 流程；在 2025-11-25 中降级为 `MAY` 回退 |
| JWKS | "用于 JWT 验证的公钥" | JSON Web Key Set，从 `jwks_uri` 获取，按 `kid` 索引 |
| 轮换 vs 刷新 | "更新密钥" | *轮换* = AS 铸造/退役签名密钥；*刷新* = 资源服务器重新获取已发布的密钥集。资源服务器只执行刷新 |
| 资源指示器 | "受众参数" | RFC 8707 `resource` 参数，将令牌绑定到一个服务器 |
| `aud` 声明 | "受众" | 验证器与规范资源 URL 比较的 JWT 声明 |
| 受众重放 | "令牌重放" | 为服务器 A 签发的令牌呈现给服务器 B；通过受众验证防御（规范：访问令牌权限限制） |
| 混淆代理 | "代理令牌滥用" | 使用静态客户端 ID 的 MCP 代理在无每客户端同意的情况下转发令牌；与受众重放不同 |
| 混淆攻击 | "错误的令牌端点" | 客户端被引导到在攻击者端点兑换诚实 AS 的授权码；通过 RFC 9207 `iss` 在客户端侧防御 |
| `iss` 允许列表 | "受信任的授权服务器" | 受保护资源元数据 `authorization_servers` 中命名的集合 |
| `resource_metadata` | "去哪找 PRM 文档" | 401/403 上 `WWW-Authenticate` 参数，命名 RFC 9728 元数据 URL |
| 公共客户端 | "原生或浏览器客户端" | 无 `client_secret` 的 OAuth 客户端；PKCE 补偿 |
| `WWW-Authenticate` | "401/403 响应头" | 携带驱动客户端恢复的 `Bearer error=...` 指令 |

## 延伸阅读

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 本课实现的 MCP 认证配置文件
- [MCP blog — One Year of MCP: November 2025 Spec Release](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 2025-11-25 的变更（CIMD、XAA、DCR 降级）
- [Aaron Parecki — Client Registration in the November 2025 MCP Authorization Spec](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) — CIMD 优于 DCR 的理由
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — 发现契约
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR（回退路径）
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端持有者证明
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 受众绑定
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — 防御混淆攻击的 `iss` 参数
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 统一的 OAuth 基底
