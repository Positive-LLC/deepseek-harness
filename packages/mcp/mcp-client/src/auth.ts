/**
 * Optional OAuth 2.1 authorization for a remote MCP server.
 *
 * The MCP SDK owns the whole protocol — protected-resource discovery, dynamic
 * client registration, PKCE, the code exchange and refresh — behind
 * `auth(provider, …)` and its `OAuthClientProvider` callbacks. This module owns
 * only the two things the SDK deliberately leaves to the application:
 *
 * 1. **Where the state lives.** {@link FileOAuthProvider} is a file-backed
 *    provider. It holds no in-memory cache: several plugin rows can deliberately
 *    share one store (one authorization covers every server the same issuer
 *    protects), so every read comes from disk and every write is a
 *    read-modify-write. Writes are atomic (temp file + rename) so a crash cannot
 *    leave a half-written token file.
 * 2. **How the operator authorizes once.** There is no browser inside the
 *    harness, so {@link authorize} runs the SDK entry point and, when it answers
 *    `REDIRECT`, serves the configured redirect URL with a one-shot local
 *    listener, hands the captured code back to the SDK, and returns once the
 *    tokens are stored. Every later connection is silent: the SDK refreshes.
 *
 * @module
 */

import { createServer } from 'node:http'
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

import { auth, type OAuthClientProvider } from '@modelcontextprotocol/sdk/client/auth.js'
import type {
  OAuthClientInformationMixed,
  OAuthClientMetadata,
  OAuthTokens,
} from '@modelcontextprotocol/sdk/shared/auth.js'

/** Resolved `auth` config for one remote server. */
export interface AuthConfig {
  /** JSON file holding the client registration, the PKCE verifier and the tokens. */
  readonly storePath: string
  /** Redirect URI the authorization server returns the code to; a local listener answers it. */
  readonly redirectUrl: string
  /** Space-separated scopes requested at authorization. */
  readonly scope: string
  /** Human-readable client name sent with dynamic registration. */
  readonly clientName: string
}

/** Durable state the SDK's provider callbacks read and write. */
interface StoredState {
  clientInformation?: OAuthClientInformationMixed
  tokens?: OAuthTokens
  codeVerifier?: string
}

/** The credential scopes the SDK may ask a provider to drop. */
export type CredentialScope = 'all' | 'client' | 'tokens' | 'verifier' | 'discovery'

/** How long the one-shot listener waits for the operator to finish the browser login. */
export const AUTHORIZATION_TIMEOUT_MS = 300_000

/**
 * A file-backed {@link OAuthClientProvider}.
 *
 * One instance per server connection generation: the SDK expects the same
 * provider object across the redirect and the code exchange. The FILE may be
 * shared by several servers, which is why nothing is cached here — a sibling
 * that authorized first has to be visible to this one on its next attempt.
 */
export class FileOAuthProvider implements OAuthClientProvider {
  /**
   * The URL `redirectToAuthorization` was handed, recorded so the caller can
   * surface it; the SDK does not return it from `auth()`.
   */
  authorizationUrl: URL | undefined

  /**
   * @param config - resolved `auth` configuration for this server.
   */
  constructor(readonly config: AuthConfig) {}

  /** The redirect URI the local listener binds. */
  get redirectUrl(): string {
    return this.config.redirectUrl
  }

  /** Metadata sent with dynamic client registration. */
  get clientMetadata(): OAuthClientMetadata {
    return {
      client_name: this.config.clientName,
      redirect_uris: [this.config.redirectUrl],
      grant_types: ['authorization_code', 'refresh_token'],
      response_types: ['code'],
      token_endpoint_auth_method: 'none',
      scope: this.config.scope,
    }
  }

  /** Read the store. A missing, unreadable or malformed file is empty state. */
  private async load(): Promise<StoredState> {
    try {
      const raw = await readFile(this.config.storePath, 'utf8')
      const parsed: unknown = JSON.parse(raw)
      return typeof parsed === 'object' && parsed !== null ? (parsed as StoredState) : {}
    } catch {
      return {}
    }
  }

  /**
   * Persist a whole state atomically, so a crash cannot leave a truncated token
   * file and a concurrent reader sees either the old or the new file.
   * @param state - the state to write.
   */
  private async persist(state: StoredState): Promise<void> {
    const path = this.config.storePath
    await mkdir(dirname(path), { recursive: true })
    const temporary = `${path}.tmp`
    await writeFile(temporary, `${JSON.stringify(state, null, 2)}\n`, { mode: 0o600 })
    await rename(temporary, path)
  }

  /** @returns the stored client registration, or undefined before the first registration. */
  async clientInformation(): Promise<OAuthClientInformationMixed | undefined> {
    return (await this.load()).clientInformation
  }

  /** @param information - registration the SDK just obtained from the server. */
  async saveClientInformation(information: OAuthClientInformationMixed): Promise<void> {
    const state = await this.load()
    state.clientInformation = information
    await this.persist(state)
  }

  /** @returns the stored tokens, or undefined when nobody has authorized yet. */
  async tokens(): Promise<OAuthTokens | undefined> {
    return (await this.load()).tokens
  }

  /** @param tokens - freshly issued or refreshed tokens. */
  async saveTokens(tokens: OAuthTokens): Promise<void> {
    const state = await this.load()
    state.tokens = tokens
    await this.persist(state)
  }

  /** @param codeVerifier - PKCE verifier to hold until the code exchange. */
  async saveCodeVerifier(codeVerifier: string): Promise<void> {
    const state = await this.load()
    state.codeVerifier = codeVerifier
    await this.persist(state)
  }

  /** @returns the PKCE verifier saved before the redirect. */
  async codeVerifier(): Promise<string> {
    const verifier = (await this.load()).codeVerifier
    if (verifier === undefined) {
      throw new Error(`mcp-client: no PKCE code verifier stored at ${this.config.storePath}`)
    }
    return verifier
  }

  /**
   * Record the authorization URL and republish it where an operator will look.
   *
   * The harness has no user agent to redirect, so this does not navigate. It
   * keeps the URL for {@link authorize} and writes it beside the store, because
   * neither a headless boot nor a GUI launch reliably shows a plugin's log line
   * to the person who has to open it.
   * @param authorizationUrl - URL the operator must visit.
   */
  async redirectToAuthorization(authorizationUrl: URL): Promise<void> {
    this.authorizationUrl = authorizationUrl
    const path = `${this.config.storePath}.authorize-url`
    await mkdir(dirname(path), { recursive: true })
    await writeFile(path, `${authorizationUrl.href}\n`, { mode: 0o600 })
  }

  /**
   * Drop credentials the server has rejected, so the next attempt re-authorizes
   * instead of replaying a token the server has already refused.
   * @param scope - which stored records to drop.
   */
  async invalidateCredentials(scope: CredentialScope): Promise<void> {
    const state = await this.load()
    if (scope === 'all' || scope === 'client') delete state.clientInformation
    if (scope === 'all' || scope === 'tokens') delete state.tokens
    if (scope === 'all' || scope === 'verifier') delete state.codeVerifier
    await this.persist(state)
  }
}

/**
 * Serve the redirect URI once and resolve with the authorization code it carries.
 *
 * Binds only the host and port the configured redirect URL names, answers one
 * request, and closes. The path is matched so an unrelated probe on the same
 * port cannot complete the flow with a stray query parameter.
 *
 * @param redirectUrl - absolute redirect URI the server will send the code to.
 * @param timeoutMs - how long to wait for the browser round trip.
 * @param signal - the connection generation's abort signal; aborting rejects.
 * @returns the authorization code from the redirect.
 */
export function waitForAuthorizationCode(
  redirectUrl: string,
  timeoutMs: number,
  signal: AbortSignal,
): Promise<string> {
  const target = new URL(redirectUrl)
  return new Promise<string>((resolve, reject) => {
    const finish = (settle: () => void): void => {
      clearTimeout(timer)
      signal.removeEventListener('abort', onAbort)
      server.close()
      settle()
    }
    const server = createServer((request, response) => {
      const requestUrl = new URL(request.url ?? '/', `${target.protocol}//${target.host}`)
      if (requestUrl.pathname !== target.pathname) {
        response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' })
        response.end('not the authorization callback path\n')
        return
      }
      const failure = requestUrl.searchParams.get('error')
      const code = requestUrl.searchParams.get('code')
      response.writeHead(200, { 'content-type': 'text/plain; charset=utf-8' })
      response.end(
        failure === null
          ? 'Authorization complete — you can close this tab and return to the harness.\n'
          : `Authorization failed: ${failure}\n`,
      )
      finish(() => {
        if (failure !== null) reject(new Error(`mcp-client: authorization failed: ${failure}`))
        else if (code === null) reject(new Error('mcp-client: authorization redirect carried no code'))
        else resolve(code)
      })
    })
    const timer = setTimeout(() => {
      finish(() => reject(new Error(`mcp-client: authorization timed out after ${timeoutMs}ms`)))
    }, timeoutMs)
    const onAbort = (): void => {
      finish(() => reject(new Error('mcp-client: authorization aborted')))
    }
    signal.addEventListener('abort', onAbort, { once: true })
    server.on('error', (error) => {
      finish(() => reject(error))
    })
    server.listen(
      Number(target.port === '' ? (target.protocol === 'https:' ? 443 : 80) : target.port),
      target.hostname,
    )
  })
}

/**
 * Authorize one server, completing the interactive leg when the SDK asks for it.
 *
 * `auth()` returns `AUTHORIZED` when stored tokens were still good (the SDK
 * refreshed them if needed). Otherwise it started the redirect and the provider
 * recorded the URL, which `onAuthorizationUrl` receives so the caller can tell
 * the operator where to go; the code comes back through a one-shot listener on
 * the configured redirect URL.
 *
 * When several servers intentionally share one store, the first to finish stores
 * tokens every sibling then finds — so this re-reads the store after the redirect
 * and skips the exchange if somebody else has already authorized.
 *
 * @param provider - the provider this generation's transport was built with.
 * @param serverUrl - the MCP server URL the flow discovers from.
 * @param signal - the connection generation's abort signal.
 * @param onAuthorizationUrl - called once with the URL the operator must open.
 * @param timeoutMs - how long to wait for the operator.
 */
export async function authorize(
  provider: FileOAuthProvider,
  serverUrl: string,
  signal: AbortSignal,
  onAuthorizationUrl: (url: URL) => void,
  timeoutMs: number = AUTHORIZATION_TIMEOUT_MS,
): Promise<void> {
  const started = await auth(provider, { serverUrl, scope: provider.config.scope })
  if (started === 'AUTHORIZED') return
  const url = provider.authorizationUrl
  if (url === undefined) {
    throw new Error('mcp-client: the server requires authorization but no authorization URL was produced')
  }
  onAuthorizationUrl(url)
  const code = await waitForAuthorizationCode(provider.config.redirectUrl, timeoutMs, signal)
  const finished = await auth(provider, { serverUrl, scope: provider.config.scope, authorizationCode: code })
  if (finished !== 'AUTHORIZED') {
    throw new Error('mcp-client: the authorization code exchange did not produce tokens')
  }
}
