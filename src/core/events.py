"""
Event System (Pub/Sub)

Provides a decoupled event-driven architecture for communication between components.
"""

from typing import Callable, List, Dict, Any, TypeVar, Generic, Optional, Union, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import threading
from collections import defaultdict


T = TypeVar('T')


@dataclass
class Event:
    """Base event class"""
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
            "correlation_id": self.correlation_id
        }


# Event type definitions
class SystemEvents:
    """System-level events"""
    STARTUP = "system.startup"
    SHUTDOWN = "system.shutdown"
    CONFIG_CHANGED = "system.config_changed"
    ERROR = "system.error"


class SessionEvents:
    """Session-related events"""
    STARTED = "session.started"
    ENDED = "session.ended"
    MODE_CHANGED = "session.mode_changed"


class MessageEvents:
    """Message-related events"""
    USER_MESSAGE = "message.user"
    AI_RESPONSE = "message.ai_response"
    TOOL_CALL = "message.tool_call"
    TOOL_RESULT = "message.tool_result"


class TaskEvents:
    """Task-related events"""
    CREATED = "task.created"
    UPDATED = "task.updated"
    COMPLETED = "task.completed"
    FAILED = "task.failed"


EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Coroutine[Any, Any, None]]


@dataclass
class Subscription:
    """Event subscription"""
    event_type: str
    handler: Union[EventHandler, AsyncEventHandler]
    priority: int = 0
    filter_func: Optional[Callable[[Event], bool]] = None


class EventBus:
    """
    Central event bus for publish/subscribe pattern.
    
    Example:
        bus = EventBus()
        
        # Subscribe to events
        def on_user_message(event: Event):
            print(f"User said: {event.data['message']}")
        
        bus.subscribe(MessageEvents.USER_MESSAGE, on_user_message)
        
        # Publish events
        bus.publish(Event(
            event_type=MessageEvents.USER_MESSAGE,
            data={"message": "Hello!"}
        ))
    """
    
    def __init__(self):
        self._subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self._wildcard_subscriptions: List[Subscription] = []
        self._lock = threading.RLock()
        self._event_history: List[Event] = []
        self._max_history = 1000
    
    def subscribe(
        self,
        event_type: str,
        handler: Union[EventHandler, AsyncEventHandler],
        priority: int = 0,
        filter_func: Optional[Callable[[Event], bool]] = None
    ) -> 'Subscription':
        """
        Subscribe to an event type.
        
        Args:
            event_type: Event type to subscribe to (use "*" for all events)
            handler: Handler function (sync or async)
            priority: Handler priority (higher = called first)
            filter_func: Optional filter function
        
        Returns:
            Subscription object (can be used to unsubscribe)
        """
        subscription = Subscription(
            event_type=event_type,
            handler=handler,
            priority=priority,
            filter_func=filter_func
        )
        
        with self._lock:
            if event_type == "*":
                self._wildcard_subscriptions.append(subscription)
                self._wildcard_subscriptions.sort(key=lambda s: -s.priority)
            else:
                self._subscriptions[event_type].append(subscription)
                self._subscriptions[event_type].sort(key=lambda s: -s.priority)
        
        return subscription
    
    def unsubscribe(self, subscription: Subscription):
        """Unsubscribe from events"""
        with self._lock:
            if subscription.event_type == "*":
                if subscription in self._wildcard_subscriptions:
                    self._wildcard_subscriptions.remove(subscription)
            else:
                if subscription.event_type in self._subscriptions:
                    subs = self._subscriptions[subscription.event_type]
                    if subscription in subs:
                        subs.remove(subscription)
    
    def publish(self, event: Event):
        """
        Publish an event synchronously.
        
        Args:
            event: Event to publish
        """
        # Add to history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
            
            # Get subscriptions
            subscriptions = self._subscriptions.get(event.event_type, []).copy()
            subscriptions.extend(self._wildcard_subscriptions)
        
        # Call handlers
        for sub in subscriptions:
            # Apply filter if present
            if sub.filter_func and not sub.filter_func(event):
                continue
            
            try:
                if asyncio.iscoroutinefunction(sub.handler):
                    # Schedule async handler
                    asyncio.create_task(sub.handler(event))
                else:
                    # Call sync handler
                    sub.handler(event)
            except Exception as e:
                # Don't let handler errors break the event bus
                self._handle_error(e, event, sub)
    
    async def publish_async(self, event: Event):
        """
        Publish an event asynchronously and wait for all handlers.
        
        Args:
            event: Event to publish
        """
        # Add to history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
            
            # Get subscriptions
            subscriptions = self._subscriptions.get(event.event_type, []).copy()
            subscriptions.extend(self._wildcard_subscriptions)
        
        # Call handlers
        tasks = []
        for sub in subscriptions:
            # Apply filter if present
            if sub.filter_func and not sub.filter_func(event):
                continue
            
            try:
                if asyncio.iscoroutinefunction(sub.handler):
                    tasks.append(sub.handler(event))
                else:
                    # Wrap sync handler in async
                    async def call_sync():
                        sub.handler(event)
                    tasks.append(call_sync())
            except Exception as e:
                self._handle_error(e, event, sub)
        
        # Wait for all handlers
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type (None = all events)
            limit: Maximum number of events to return
        
        Returns:
            List of events (most recent first)
        """
        with self._lock:
            history = self._event_history.copy()
        
        if event_type:
            history = [e for e in history if e.event_type == event_type]
        
        # Return most recent first
        history.reverse()
        return history[:limit]
    
    def clear_history(self):
        """Clear event history"""
        with self._lock:
            self._event_history.clear()
    
    def _handle_error(self, error: Exception, event: Event, subscription: Subscription):
        """Handle handler errors"""
        # Publish error event (but don't create infinite loop)
        if event.event_type != SystemEvents.ERROR:
            error_event = Event(
                event_type=SystemEvents.ERROR,
                source="event_bus",
                data={
                    "error": str(error),
                    "original_event": event.to_dict(),
                    "handler": str(subscription.handler)
                }
            )
            # Add to history only (don't re-publish)
            with self._lock:
                self._event_history.append(error_event)


# Global event bus instance
_global_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Get the global event bus"""
    return _global_event_bus


def subscribe(
    event_type: str,
    priority: int = 0,
    filter_func: Optional[Callable[[Event], bool]] = None
):
    """
    Decorator for subscribing to events.
    
    Example:
        @subscribe(MessageEvents.USER_MESSAGE)
        def on_user_message(event: Event):
            print(f"Message: {event.data['message']}")
    """
    def decorator(func: Callable) -> Callable:
        _global_event_bus.subscribe(
            event_type=event_type,
            handler=func,
            priority=priority,
            filter_func=filter_func
        )
        return func
    return decorator


def publish(event_type: str, **data):
    """
    Publish an event to the global bus.
    
    Example:
        publish(MessageEvents.USER_MESSAGE, message="Hello", user_id="123")
    """
    event = Event(event_type=event_type, data=data)
    _global_event_bus.publish(event)


async def publish_async(event_type: str, **data):
    """Async version of publish"""
    event = Event(event_type=event_type, data=data)
    await _global_event_bus.publish_async(event)


class EventAggregator:
    """
    Aggregates events over a time window.
    
    Example:
        aggregator = EventAggregator(
            event_type=MessageEvents.TOOL_CALL,
            window_size=60.0,  # 1 minute
            on_aggregate=lambda events: print(f"{len(events)} tool calls in last minute")
        )
    """
    
    def __init__(
        self,
        event_type: str,
        window_size: float,  # seconds
        on_aggregate: Callable[[List[Event]], None]
    ):
        self.event_type = event_type
        self.window_size = window_size
        self.on_aggregate = on_aggregate
        self._events: List[Event] = []
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        
        # Subscribe to events
        get_event_bus().subscribe(event_type, self._on_event)
        self._start_timer()
    
    def _on_event(self, event: Event):
        """Handle incoming event"""
        with self._lock:
            self._events.append(event)
    
    def _start_timer(self):
        """Start aggregation timer"""
        def trigger():
            with self._lock:
                if self._events:
                    # Call aggregation callback
                    try:
                        self.on_aggregate(self._events.copy())
                    except Exception:
                        pass
                    
                    # Clear events
                    self._events.clear()
            
            # Restart timer
            self._start_timer()
        
        self._timer = threading.Timer(self.window_size, trigger)
        self._timer.daemon = True
        self._timer.start()
    
    def stop(self):
        """Stop the aggregator"""
        if self._timer:
            self._timer.cancel()
