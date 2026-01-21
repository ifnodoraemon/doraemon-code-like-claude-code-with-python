"""
Unit tests for the Event System (Pub/Sub).

Tests event publishing, subscription, filtering, and aggregation.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from src.core.events import (
    Event,
    EventAggregator,
    EventBus,
    MessageEvents,
    SessionEvents,
    SystemEvents,
    TaskEvents,
    get_event_bus,
    publish,
    subscribe,
)


class TestEvent:
    """Tests for Event class"""

    def test_event_creation(self):
        """Test creating an event with default values"""
        event = Event(event_type="test.event")

        assert event.event_type == "test.event"
        assert event.timestamp is not None
        assert event.data == {}
        assert event.source is None
        assert event.correlation_id is None

    def test_event_with_data(self):
        """Test creating an event with data"""
        event = Event(
            event_type="user.created",
            data={"user_id": "123", "email": "test@example.com"},
            source="user_service",
            correlation_id="req-456"
        )

        assert event.event_type == "user.created"
        assert event.data["user_id"] == "123"
        assert event.source == "user_service"
        assert event.correlation_id == "req-456"

    def test_event_to_dict(self):
        """Test converting event to dictionary"""
        event = Event(
            event_type="test.event",
            data={"key": "value"},
            source="test",
            correlation_id="123"
        )

        d = event.to_dict()

        assert d["event_type"] == "test.event"
        assert d["data"] == {"key": "value"}
        assert d["source"] == "test"
        assert d["correlation_id"] == "123"
        assert "timestamp" in d


class TestEventBus:
    """Tests for EventBus class"""

    def test_subscribe_and_publish(self):
        """Test basic subscribe and publish"""
        bus = EventBus()
        received = []

        def handler(event: Event):
            received.append(event)

        bus.subscribe("test.event", handler)
        bus.publish(Event(event_type="test.event", data={"value": 42}))

        assert len(received) == 1
        assert received[0].data["value"] == 42

    def test_multiple_subscribers(self):
        """Test multiple subscribers to same event type"""
        bus = EventBus()
        received1 = []
        received2 = []

        bus.subscribe("test.event", lambda e: received1.append(e))
        bus.subscribe("test.event", lambda e: received2.append(e))

        bus.publish(Event(event_type="test.event"))

        assert len(received1) == 1
        assert len(received2) == 1

    def test_wildcard_subscription(self):
        """Test wildcard subscription receives all events"""
        bus = EventBus()
        received = []

        bus.subscribe("*", lambda e: received.append(e))

        bus.publish(Event(event_type="event.one"))
        bus.publish(Event(event_type="event.two"))
        bus.publish(Event(event_type="event.three"))

        assert len(received) == 3

    def test_subscription_priority(self):
        """Test that higher priority handlers are called first"""
        bus = EventBus()
        order = []

        bus.subscribe("test", lambda e: order.append("low"), priority=0)
        bus.subscribe("test", lambda e: order.append("high"), priority=10)
        bus.subscribe("test", lambda e: order.append("medium"), priority=5)

        bus.publish(Event(event_type="test"))

        assert order == ["high", "medium", "low"]

    def test_subscription_filter(self):
        """Test event filtering"""
        bus = EventBus()
        received = []

        # Only receive events where data["important"] is True
        bus.subscribe(
            "test",
            lambda e: received.append(e),
            filter_func=lambda e: e.data.get("important", False)
        )

        bus.publish(Event(event_type="test", data={"important": False}))
        bus.publish(Event(event_type="test", data={"important": True}))
        bus.publish(Event(event_type="test", data={}))

        assert len(received) == 1
        assert received[0].data["important"] is True

    def test_unsubscribe(self):
        """Test unsubscribing from events"""
        bus = EventBus()
        received = []

        subscription = bus.subscribe("test", lambda e: received.append(e))

        bus.publish(Event(event_type="test"))
        assert len(received) == 1

        bus.unsubscribe(subscription)

        bus.publish(Event(event_type="test"))
        assert len(received) == 1  # Should not increase

    def test_event_history(self):
        """Test event history tracking"""
        bus = EventBus()

        bus.publish(Event(event_type="event.one"))
        bus.publish(Event(event_type="event.two"))
        bus.publish(Event(event_type="event.three"))

        history = bus.get_history()

        assert len(history) == 3
        # Most recent first
        assert history[0].event_type == "event.three"

    def test_event_history_filtered(self):
        """Test filtered event history"""
        bus = EventBus()

        bus.publish(Event(event_type="type.a"))
        bus.publish(Event(event_type="type.b"))
        bus.publish(Event(event_type="type.a"))

        history = bus.get_history(event_type="type.a")

        assert len(history) == 2
        assert all(e.event_type == "type.a" for e in history)

    def test_event_history_limit(self):
        """Test event history limit"""
        bus = EventBus()

        for i in range(10):
            bus.publish(Event(event_type="test", data={"index": i}))

        history = bus.get_history(limit=5)

        assert len(history) == 5
        # Most recent first
        assert history[0].data["index"] == 9

    def test_clear_history(self):
        """Test clearing event history"""
        bus = EventBus()

        bus.publish(Event(event_type="test"))
        bus.publish(Event(event_type="test"))

        assert len(bus.get_history()) == 2

        bus.clear_history()

        assert len(bus.get_history()) == 0

    def test_handler_error_isolation(self):
        """Test that handler errors don't break other handlers"""
        bus = EventBus()
        received = []

        def bad_handler(event):
            raise Exception("Handler error")

        def good_handler(event):
            received.append(event)

        bus.subscribe("test", bad_handler, priority=10)
        bus.subscribe("test", good_handler, priority=0)

        # Should not raise
        bus.publish(Event(event_type="test"))

        # Good handler should still receive event
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_publish_async(self):
        """Test async event publishing"""
        bus = EventBus()
        received = []

        async def async_handler(event: Event):
            await asyncio.sleep(0.01)
            received.append(event)

        bus.subscribe("test", async_handler)

        await bus.publish_async(Event(event_type="test"))

        assert len(received) == 1


class TestEventTypes:
    """Tests for predefined event types"""

    def test_system_events(self):
        """Test system event type constants"""
        assert SystemEvents.STARTUP == "system.startup"
        assert SystemEvents.SHUTDOWN == "system.shutdown"
        assert SystemEvents.ERROR == "system.error"

    def test_session_events(self):
        """Test session event type constants"""
        assert SessionEvents.STARTED == "session.started"
        assert SessionEvents.ENDED == "session.ended"
        assert SessionEvents.MODE_CHANGED == "session.mode_changed"

    def test_message_events(self):
        """Test message event type constants"""
        assert MessageEvents.USER_MESSAGE == "message.user"
        assert MessageEvents.TOOL_CALL == "message.tool_call"
        assert MessageEvents.TOOL_RESULT == "message.tool_result"

    def test_task_events(self):
        """Test task event type constants"""
        assert TaskEvents.CREATED == "task.created"
        assert TaskEvents.COMPLETED == "task.completed"
        assert TaskEvents.FAILED == "task.failed"


class TestGlobalFunctions:
    """Tests for global event functions"""

    def test_get_event_bus(self):
        """Test getting global event bus"""
        bus = get_event_bus()

        assert isinstance(bus, EventBus)

        # Should return same instance
        bus2 = get_event_bus()
        assert bus is bus2

    def test_subscribe_decorator(self):
        """Test @subscribe decorator"""
        received = []

        @subscribe("decorator.test")
        def handler(event: Event):
            received.append(event)

        # Publish to global bus
        publish("decorator.test", data="test")

        assert len(received) == 1
        assert received[0].data["data"] == "test"

    def test_publish_function(self):
        """Test publish helper function"""
        bus = get_event_bus()
        received = []

        bus.subscribe("publish.test", lambda e: received.append(e))

        publish("publish.test", key1="value1", key2="value2")

        assert len(received) == 1
        assert received[0].data["key1"] == "value1"
        assert received[0].data["key2"] == "value2"


class TestEventAggregator:
    """Tests for EventAggregator class"""

    def test_aggregator_collects_events(self):
        """Test that aggregator collects events"""
        aggregated = []

        def on_aggregate(events):
            aggregated.append(events)

        aggregator = EventAggregator(
            event_type="agg.test",
            window_size=0.1,  # 100ms window
            on_aggregate=on_aggregate
        )

        try:
            bus = get_event_bus()

            # Publish some events
            bus.publish(Event(event_type="agg.test", data={"n": 1}))
            bus.publish(Event(event_type="agg.test", data={"n": 2}))

            # Wait for aggregation
            time.sleep(0.15)

            assert len(aggregated) >= 1
            # First aggregation should have 2 events
            assert len(aggregated[0]) == 2
        finally:
            aggregator.stop()

    def test_aggregator_stop(self):
        """Test stopping aggregator"""
        callback = MagicMock()

        aggregator = EventAggregator(
            event_type="stop.test",
            window_size=0.5,  # Longer window to ensure timer is running
            on_aggregate=callback
        )

        # Stop the aggregator
        aggregator.stop()

        # Give time for cancellation to take effect
        time.sleep(0.05)

        # Timer should be cancelled (may still exist but should be cancelled)
        # After cancel(), the timer won't fire even if still "alive" briefly
        if aggregator._timer is not None:
            # Timer was cancelled, it shouldn't fire the callback
            time.sleep(0.6)  # Wait past the window
            # Callback should not have been called after stop
            callback.assert_not_called()
