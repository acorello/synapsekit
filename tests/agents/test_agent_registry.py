import pytest

from synapsekit.agents import (
    AgentMetadata,
    AgentRegistry,
    InMemoryAgentRegistry,
    RedisAgentRegistry,
)


class ManualClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_in_memory_registry_registers_and_discovers_agents() -> None:
    clock = ManualClock()
    registry = InMemoryAgentRegistry(stale_timeout=10, clock=clock)

    registry.register(
        AgentMetadata(
            id="research-1",
            model="gpt-4o",
            tools=["search", "summarize"],
            capacity=3,
            cost_multiplier=1.4,
            tags=["research", "premium"],
        )
    )
    registry.register(
        AgentMetadata(
            id="support-1",
            model="gpt-4o-mini",
            tools=["ticket"],
            capacity=5,
            cost_multiplier=0.7,
            tags=["support"],
        )
    )

    discovered = registry.discover(tools=["search"], tags=["research"], min_capacity=2)

    assert [agent.id for agent in discovered] == ["research-1"]
    assert registry.get("research-1").last_heartbeat == 100.0


def test_heartbeat_refreshes_health_and_stale_agents_are_pruned() -> None:
    clock = ManualClock()
    registry = InMemoryAgentRegistry(stale_timeout=5, clock=clock)
    registry.register(AgentMetadata(id="stale", model="cheap", last_heartbeat=90))
    registry.register(AgentMetadata(id="fresh", model="fast"))

    assert registry.is_healthy("stale") is False
    assert registry.is_healthy("fresh") is True

    registry.heartbeat("stale")
    assert registry.is_healthy("stale") is True

    clock.value = 107
    assert registry.prune_stale() == ["stale", "fresh"]
    assert registry.list() == []


def test_agent_registry_facade_uses_in_memory_backend() -> None:
    registry = AgentRegistry(stale_timeout=1)
    registry.register(AgentMetadata(id="agent", model="gpt-4o-mini"))

    assert registry.get("agent").id == "agent"
    assert registry.discover(healthy_only=True)[0].id == "agent"


def test_unknown_heartbeat_raises_key_error() -> None:
    registry = InMemoryAgentRegistry()

    with pytest.raises(KeyError):
        registry.heartbeat("missing")


class FakeRedis:
    def __init__(self) -> None:
        self.values = {}

    def set(self, key, value):
        self.values[key] = value
        return True

    def get(self, key):
        return self.values.get(key)

    def delete(self, key):
        return int(self.values.pop(key, None) is not None)

    def scan_iter(self, match):
        prefix = match[:-1]
        for key in list(self.values):
            if key.startswith(prefix):
                yield key


def test_redis_registry_works_with_injected_client_without_redis_dependency() -> None:
    registry = RedisAgentRegistry(redis_client=FakeRedis(), stale_timeout=10)
    registry.register(
        AgentMetadata(
            id="redis-agent",
            model="gpt-4o-mini",
            tools=["ticket"],
            tags=["support"],
        )
    )

    assert registry.get("redis-agent").id == "redis-agent"
    assert registry.discover(tools="ticket", tags="support")[0].id == "redis-agent"

    registry.heartbeat("redis-agent", timestamp=1)
    assert registry.prune_stale() == ["redis-agent"]
