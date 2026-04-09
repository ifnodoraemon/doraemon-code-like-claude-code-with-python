import test from 'node:test'
import assert from 'node:assert/strict'

import {
    filterMessagesForRange,
    getNextSessionIndex,
    isMessageRangeLoaded,
} from '../src/lib/sessionWindow.ts'

test('filterMessagesForRange uses absolute session indexes for run windows', () => {
    const messages = [
        { session_index: 10, content: 'u' },
        { session_index: 11, content: 'a' },
        { session_index: 12, content: 'u2' },
    ]

    const visible = filterMessagesForRange(messages, {
        message_start_index: 10,
        message_end_index: 11,
    })

    assert.deepEqual(
        visible.map((message) => message.content),
        ['u', 'a'],
    )
})

test('getNextSessionIndex appends after the last known absolute session index', () => {
    const messages = [
        { session_index: 98 },
        { session_index: 99 },
        { session_index: 100 },
    ]

    assert.equal(getNextSessionIndex(messages, 0), 101)
})

test('getNextSessionIndex falls back to window offset when messages are not indexed', () => {
    const messages = [{ content: 'u' }, { content: 'a' }]

    assert.equal(getNextSessionIndex(messages, 40), 42)
})

test('isMessageRangeLoaded checks indexed window bounds before using fallback offsets', () => {
    const messages = [
        { session_index: 100, content: 'u' },
        { session_index: 101, content: 'a' },
        { session_index: 102, content: 'u2' },
    ]

    assert.equal(isMessageRangeLoaded(100, 102, 0, messages), true)
    assert.equal(isMessageRangeLoaded(99, 102, 0, messages), false)
    assert.equal(isMessageRangeLoaded(100, 103, 0, messages), false)
})
