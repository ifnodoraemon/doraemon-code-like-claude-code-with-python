import test from 'node:test'
import assert from 'node:assert/strict'

import {
    AUTH_REQUIRED_EVENT,
    clearStoredApiKey,
    installApiFetch,
    isProtectedApiRequest,
    setStoredApiKey,
    withApiAuth,
} from '../src/apiAuth.ts'

function installBrowserWindow(fetchImpl = async () => new Response(null, { status: 204 })) {
    const previousWindow = globalThis.window
    const previousCustomEvent = globalThis.CustomEvent
    const storage = new Map()
    const dispatched = []

    globalThis.CustomEvent = class CustomEvent extends Event {
        constructor(type, options = {}) {
            super(type)
            this.detail = options.detail
        }
    }

    globalThis.window = {
        location: new URL('http://127.0.0.1:3000/'),
        fetch: fetchImpl,
        localStorage: {
            getItem: (key) => storage.get(key) ?? null,
            setItem: (key, value) => storage.set(key, value),
            removeItem: (key) => storage.delete(key),
        },
        addEventListener: () => {},
        removeEventListener: () => {},
        dispatchEvent: (event) => {
            dispatched.push(event)
            return true
        },
    }

    return {
        dispatched,
        restore: () => {
            globalThis.window = previousWindow
            globalThis.CustomEvent = previousCustomEvent
        },
    }
}

test('isProtectedApiRequest only matches same-origin API routes', () => {
    const browser = installBrowserWindow()
    try {
        assert.equal(isProtectedApiRequest('/api/sessions'), true)
        assert.equal(isProtectedApiRequest('/dashboard/api/status'), true)
        assert.equal(isProtectedApiRequest('/assets/app.js'), false)
        assert.equal(isProtectedApiRequest('http://example.com/api/sessions'), false)
    } finally {
        browser.restore()
    }
})

test('withApiAuth attaches a stored bearer token without replacing caller auth', () => {
    const browser = installBrowserWindow()
    try {
        setStoredApiKey(' secret ')
        const [, init] = withApiAuth('/api/chat', {
            headers: { 'Content-Type': 'application/json' },
        })
        const headers = new Headers(init?.headers)
        assert.equal(headers.get('Authorization'), 'Bearer secret')
        assert.equal(headers.get('Content-Type'), 'application/json')

        const [, existingAuthInit] = withApiAuth('/api/chat', {
            headers: { Authorization: 'Bearer caller' },
        })
        assert.equal(new Headers(existingAuthInit?.headers).get('Authorization'), 'Bearer caller')
    } finally {
        clearStoredApiKey()
        browser.restore()
    }
})

test('installApiFetch emits auth-required for protected API auth failures', async () => {
    const browser = installBrowserWindow(async () => new Response(null, { status: 401 }))
    try {
        installApiFetch()
        const response = await globalThis.window.fetch('/api/sessions')

        assert.equal(response.status, 401)
        assert.equal(browser.dispatched.length, 1)
        assert.equal(browser.dispatched[0].type, AUTH_REQUIRED_EVENT)
        assert.deepEqual(browser.dispatched[0].detail, { status: 401 })
    } finally {
        browser.restore()
    }
})
