import asyncio

from synapsekit.agents import (
    AgentFederation,
    AgentMetadata,
    InMemoryAgentRegistry,
    LocalAgentClient,
    RoutingStrategy,
)


class RecordingClient:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    async def run(self, prompt: str, **kwargs):
        return {"agent_id": self.agent_id, "prompt": prompt, "kwargs": kwargs}


def build_federation() -> AgentFederation:
    registry = InMemoryAgentRegistry(stale_timeout=30)
    federation = AgentFederation(registry)
    agents = [
        AgentMetadata(
            id="research-a",
            model="gpt-4o",
            tools=["search", "summarize"],
            capacity=2,
            cost_multiplier=1.5,
            tags=["research"],
        ),
        AgentMetadata(
            id="research-b",
            model="gpt-4o-mini",
            tools=["search"],
            capacity=4,
            cost_multiplier=0.9,
            tags=["research"],
        ),
        AgentMetadata(
            id="compliance-a",
            model="gpt-4o",
            tools=["policy_check"],
            capacity=1,
            cost_multiplier=2.0,
            tags=["compliance"],
        ),
        AgentMetadata(
            id="support-a",
            model="gpt-4o-mini",
            tools=["ticket", "draft_reply"],
            capacity=5,
            cost_multiplier=0.7,
            tags=["support"],
        ),
        AgentMetadata(
            id="support-b",
            model="gpt-4o-mini",
            tools=["ticket"],
            capacity=3,
            cost_multiplier=0.5,
            tags=["support"],
        ),
    ]
    for agent in agents:
        federation.register_agent(agent, client=RecordingClient(agent.id))
    return federation


def test_round_robin_routes_across_five_agents() -> None:
    federation = build_federation()

    selected = [federation.select_agent(strategy=RoutingStrategy.ROUND_ROBIN).id for _ in range(7)]

    assert selected == [
        "research-a",
        "research-b",
        "compliance-a",
        "support-a",
        "support-b",
        "research-a",
        "research-b",
    ]


def test_capacity_aware_routing_prefers_highest_capacity_agent() -> None:
    federation = build_federation()

    selected = federation.select_agent(strategy="capacity_aware").id

    assert selected == "support-a"


def test_cost_aware_routing_prefers_lowest_cost_agent() -> None:
    federation = build_federation()

    selected = federation.select_agent(strategy="cost_aware").id

    assert selected == "support-b"


def test_discovery_filters_apply_before_routing() -> None:
    federation = build_federation()

    selected = federation.select_agent(
        strategy="cost_aware",
        tools=["search"],
        tags=["research"],
        min_capacity=3,
    )

    assert selected.id == "research-b"


def test_run_uses_custom_client_mapping() -> None:
    federation = build_federation()

    result = asyncio.run(federation.run("open ticket", agent_id="support-a", priority="high"))

    assert result == {
        "agent_id": "support-a",
        "prompt": "open ticket",
        "kwargs": {"priority": "high"},
    }


def test_local_agent_client_wraps_sync_executor() -> None:
    class Executor:
        def run(self, prompt: str, **kwargs):
            return f"{prompt}:{kwargs['suffix']}"

    client = LocalAgentClient(Executor())

    assert asyncio.run(client.run("hello", suffix="done")) == "hello:done"
