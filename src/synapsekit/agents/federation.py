"""Federated agent routing and client adapters."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from enum import Enum
from typing import Any, Protocol

from .agent_registry import AgentMetadata, AgentRegistry, InMemoryAgentRegistry


class AgentClient(Protocol):
    """Interface used by AgentFederation to invoke an agent."""

    def run(self, prompt: str, **kwargs: Any) -> Any:
        """Run the agent for a prompt."""


class LocalAgentClient:
    """AgentClient adapter for local AgentExecutor-like objects or callables."""

    def __init__(self, executor: Any) -> None:
        self.executor = executor

    async def run(self, prompt: str, **kwargs: Any) -> Any:
        target = self.executor
        if callable(target) and not any(
            hasattr(target, name)
            for name in ("arun", "run", "aexecute", "execute", "ainvoke", "invoke")
        ):
            result = target(prompt, **kwargs)
        else:
            for method_name in ("arun", "run", "aexecute", "execute", "ainvoke", "invoke"):
                method = getattr(target, method_name, None)
                if method is not None:
                    result = method(prompt, **kwargs)
                    break
            else:
                raise TypeError(
                    "LocalAgentClient requires an executor with run/arun, "
                    "execute/aexecute, invoke/ainvoke, or a callable executor."
                )

        if inspect.isawaitable(result):
            return await result
        return result


class RoutingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    CAPACITY_AWARE = "capacity_aware"
    COST_AWARE = "cost_aware"


AgentRoutingStrategy = RoutingStrategy


class AgentFederation:
    """Discover agents and route prompts across a registry."""

    def __init__(
        self,
        registry: AgentRegistry | InMemoryAgentRegistry | None = None,
        *,
        clients: dict[str, Any] | None = None,
        default_strategy: RoutingStrategy | str = RoutingStrategy.ROUND_ROBIN,
    ) -> None:
        self.registry = registry or InMemoryAgentRegistry()
        self.clients: dict[str, AgentClient] = {
            str(agent_id): self._coerce_client(client)
            for agent_id, client in (clients or {}).items()
        }
        self.default_strategy = self._normalise_strategy(default_strategy)
        self._round_robin_offsets: dict[tuple[str, tuple[str, ...]], int] = {}

    def register_agent(
        self,
        agent: AgentMetadata,
        *,
        client: Any | None = None,
    ) -> AgentMetadata:
        registered = self.registry.register(agent)
        if client is not None:
            self.clients[registered.id] = self._coerce_client(client)
        return registered

    register = register_agent

    def unregister_agent(self, agent_id: str) -> bool:
        self.clients.pop(str(agent_id), None)
        return self.registry.unregister(agent_id)

    unregister = unregister_agent

    def add_client(self, agent_id: str, client: Any) -> None:
        self.clients[str(agent_id)] = self._coerce_client(client)

    def discover(
        self,
        *,
        tools: Iterable[str] | str | None = None,
        tags: Iterable[str] | str | None = None,
        min_capacity: int | None = None,
        healthy_only: bool = True,
    ) -> list[AgentMetadata]:
        return self.registry.discover(
            tools=tools,
            tags=tags,
            min_capacity=min_capacity,
            healthy_only=healthy_only,
        )

    discover_agents = discover

    def select_agent(
        self,
        *,
        strategy: RoutingStrategy | str | None = None,
        tools: Iterable[str] | str | None = None,
        tags: Iterable[str] | str | None = None,
        min_capacity: int | None = None,
        healthy_only: bool = True,
    ) -> AgentMetadata:
        candidates = self.discover(
            tools=tools,
            tags=tags,
            min_capacity=min_capacity,
            healthy_only=healthy_only,
        )
        if not candidates:
            raise LookupError("No agents match the discovery filters.")

        selected_strategy = self._normalise_strategy(strategy or self.default_strategy)
        if selected_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin(candidates, "round_robin")
        if selected_strategy == RoutingStrategy.CAPACITY_AWARE:
            max_capacity = max(agent.capacity for agent in candidates)
            best = [agent for agent in candidates if agent.capacity == max_capacity]
            return self._round_robin(best, "capacity_aware")
        if selected_strategy == RoutingStrategy.COST_AWARE:
            min_cost = min(agent.cost_multiplier for agent in candidates)
            best = [agent for agent in candidates if agent.cost_multiplier == min_cost]
            return self._round_robin(best, "cost_aware")

        raise ValueError(f"Unsupported routing strategy: {selected_strategy}")

    route = select_agent
    select = select_agent

    async def run(
        self,
        prompt: str,
        *,
        agent_id: str | None = None,
        strategy: RoutingStrategy | str | None = None,
        tools: Iterable[str] | str | None = None,
        tags: Iterable[str] | str | None = None,
        min_capacity: int | None = None,
        healthy_only: bool = True,
        **kwargs: Any,
    ) -> Any:
        if agent_id is None:
            agent = self.select_agent(
                strategy=strategy,
                tools=tools,
                tags=tags,
                min_capacity=min_capacity,
                healthy_only=healthy_only,
            )
        else:
            agent = self.registry.get(agent_id)
            if agent is None:
                raise KeyError(f"Unknown agent id: {agent_id}")
            if healthy_only and not self.registry.is_healthy(agent.id):
                raise LookupError(f"Agent is not healthy: {agent.id}")

        client = self.get_client(agent)
        result = client.run(prompt, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def execute(self, prompt: str, **kwargs: Any) -> Any:
        return await self.run(prompt, **kwargs)

    def get_client(self, agent: AgentMetadata | str) -> AgentClient:
        agent_id = agent if isinstance(agent, str) else agent.id
        client = self.clients.get(str(agent_id))
        if client is None and not isinstance(agent, str) and agent.endpoint is not None:
            client = self.clients.get(agent.endpoint)
        if client is None:
            raise KeyError(f"No AgentClient registered for agent: {agent_id}")
        return client

    def _round_robin(self, candidates: list[AgentMetadata], strategy_key: str) -> AgentMetadata:
        ids = tuple(agent.id for agent in candidates)
        key = (strategy_key, ids)
        offset = self._round_robin_offsets.get(key, 0)
        self._round_robin_offsets[key] = offset + 1
        return candidates[offset % len(candidates)]

    def _normalise_strategy(self, strategy: RoutingStrategy | str) -> RoutingStrategy:
        if isinstance(strategy, RoutingStrategy):
            return strategy
        value = str(strategy).replace("-", "_").lower()
        aliases = {
            "roundrobin": "round_robin",
            "rr": "round_robin",
            "capacity": "capacity_aware",
            "capacity_aware_routing": "capacity_aware",
            "cost": "cost_aware",
            "cost_aware_routing": "cost_aware",
        }
        value = aliases.get(value, value)
        try:
            return RoutingStrategy(value)
        except ValueError as exc:
            raise ValueError(f"Unknown routing strategy: {strategy}") from exc

    def _coerce_client(self, client: Any) -> AgentClient:
        if hasattr(client, "run"):
            return client
        return LocalAgentClient(client)
