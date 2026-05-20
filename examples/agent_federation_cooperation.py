"""Research, compliance, and support agents cooperating through federation."""

from __future__ import annotations

import asyncio

from synapsekit.agents import (
    AgentFederation,
    AgentMetadata,
    LocalAgentClient,
    RoutingStrategy,
)


def research_agent(prompt: str) -> str:
    return f"Research brief: market context and source notes for '{prompt}'."


def compliance_agent(prompt: str) -> str:
    return f"Compliance review: checked policy and disclosure risk for '{prompt}'."


def support_agent(prompt: str) -> str:
    return f"Support response: customer-ready answer based on '{prompt}'."


async def main() -> None:
    federation = AgentFederation(default_strategy=RoutingStrategy.CAPACITY_AWARE)

    federation.register_agent(
        AgentMetadata(
            id="research",
            model="gpt-4o",
            tools=["web_search", "summarize"],
            capacity=3,
            cost_multiplier=1.5,
            tags=["research"],
        ),
        client=LocalAgentClient(research_agent),
    )
    federation.register_agent(
        AgentMetadata(
            id="compliance",
            model="gpt-4o-mini",
            tools=["policy_check"],
            capacity=2,
            cost_multiplier=1.1,
            tags=["compliance"],
        ),
        client=LocalAgentClient(compliance_agent),
    )
    federation.register_agent(
        AgentMetadata(
            id="support",
            model="gpt-4o-mini",
            tools=["ticket_lookup", "draft_reply"],
            capacity=5,
            cost_multiplier=0.8,
            tags=["support"],
        ),
        client=LocalAgentClient(support_agent),
    )

    question = "Can we promise same-day rollout for enterprise customers?"
    research = await federation.run(question, tags=["research"])
    compliance = await federation.run(research, tags=["compliance"])
    response = await federation.run(
        f"{research}\n{compliance}\nDraft a customer reply.",
        tags=["support"],
        strategy=RoutingStrategy.COST_AWARE,
    )

    print(research)
    print(compliance)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
