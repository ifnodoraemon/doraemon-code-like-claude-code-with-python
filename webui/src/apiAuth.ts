const API_KEY_STORAGE_KEY = 'doraemon.webui.apiKey'

export const AUTH_REQUIRED_EVENT = 'doraemon:auth-required'

export interface AuthRequiredDetail {
    status: number
}

function getWindow(): Window | null {
    return typeof window === 'undefined' ? null : window
}

function getRequestUrl(input: RequestInfo | URL): URL | null {
    const browserWindow = getWindow()
    const baseUrl = browserWindow?.location?.origin || 'http://localhost'

    try {
        if (typeof input === 'string') {
            return new URL(input, baseUrl)
        }
        if (input instanceof URL) {
            return input
        }
        if (typeof Request !== 'undefined' && input instanceof Request) {
            return new URL(input.url, baseUrl)
        }
    } catch {
        return null
    }

    return null
}

export function isProtectedApiRequest(input: RequestInfo | URL): boolean {
    const browserWindow = getWindow()
    if (!browserWindow?.location?.origin) {
        return false
    }

    const url = getRequestUrl(input)
    if (!url || url.origin !== browserWindow.location.origin) {
        return false
    }

    return (
        url.pathname === '/api' ||
        url.pathname.startsWith('/api/') ||
        url.pathname === '/dashboard/api' ||
        url.pathname.startsWith('/dashboard/api/')
    )
}

export function getStoredApiKey(): string {
    try {
        return getWindow()?.localStorage.getItem(API_KEY_STORAGE_KEY) || ''
    } catch {
        return ''
    }
}

export function setStoredApiKey(apiKey: string): void {
    const browserWindow = getWindow()
    if (!browserWindow) {
        return
    }

    try {
        const normalizedKey = apiKey.trim()
        if (normalizedKey) {
            browserWindow.localStorage.setItem(API_KEY_STORAGE_KEY, normalizedKey)
        } else {
            browserWindow.localStorage.removeItem(API_KEY_STORAGE_KEY)
        }
    } catch {
        // Ignore storage failures; the current request will still surface auth errors.
    }
}

export function clearStoredApiKey(): void {
    setStoredApiKey('')
}

export function isAuthFailure(response: Response): boolean {
    return response.status === 401 || response.status === 503
}

export function withApiAuth(
    input: RequestInfo | URL,
    init?: RequestInit,
): [RequestInfo | URL, RequestInit | undefined] {
    const apiKey = isProtectedApiRequest(input) ? getStoredApiKey() : ''
    if (!apiKey) {
        return [input, init]
    }

    const headers = new Headers(
        typeof Request !== 'undefined' && input instanceof Request ? input.headers : undefined,
    )
    new Headers(init?.headers).forEach((value, key) => headers.set(key, value))

    if (!headers.has('Authorization')) {
        headers.set('Authorization', `Bearer ${apiKey}`)
    }

    return [input, { ...init, headers }]
}

let fetchInstalled = false

export function installApiFetch(): void {
    const browserWindow = getWindow()
    if (!browserWindow || fetchInstalled) {
        return
    }

    fetchInstalled = true
    const originalFetch = browserWindow.fetch.bind(browserWindow)

    browserWindow.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
        const protectedRequest = isProtectedApiRequest(input)
        const [nextInput, nextInit] = withApiAuth(input, init)
        const response = await originalFetch(nextInput, nextInit)

        if (protectedRequest && isAuthFailure(response)) {
            browserWindow.dispatchEvent(
                new CustomEvent<AuthRequiredDetail>(AUTH_REQUIRED_EVENT, {
                    detail: { status: response.status },
                }),
            )
        }

        return response
    }) as typeof browserWindow.fetch
}
