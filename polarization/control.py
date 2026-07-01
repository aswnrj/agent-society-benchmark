import asyncio

from agentsociety.cityagent import (
    SocietyAgent,
    default,
)
from agentsociety.configs import (
    AgentsConfig,
    Config,
    EnvConfig,
    ExpConfig,
    LLMConfig,
    MapConfig,
)
from agentsociety.configs.agent import AgentConfig
from agentsociety.configs.exp import (
    AgentFilterConfig,
    WorkflowStepConfig,
    WorkflowType,
)
from agentsociety.environment import EnvironmentConfig
from agentsociety.llm import LLMProviderType
from agentsociety.simulation import AgentSociety
from agentsociety.storage import DatabaseConfig


config = Config(
    llm=[
        LLMConfig(
            provider=LLMProviderType.VLLM,
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            model="Qwen/Qwen2.5-14B-Instruct",
            concurrency=200,
            timeout=60,
        )
    ],
    env=EnvConfig(
        db=DatabaseConfig(
            enabled=True,
            db_type="sqlite",
            pg_dsn=None,
        ),
    ),
    map=MapConfig(
        file_path="../map.pb",
    ),
    agents=AgentsConfig(
        citizens=[
            AgentConfig(
                agent_class="citizen",
                number=100,
                memory_from_file="./profiles/profiles.json",
            )
        ],
    ),  
    exp=ExpConfig(
        name="polarization_control",
        workflow=[
            WorkflowStepConfig(
                type=WorkflowType.SAVE_CONTEXT,
                target_agent=AgentFilterConfig(
                    agent_class=(SocietyAgent,),
                ),
                key="attitude",
                save_as="guncontrol_attitude_initial",
            ),
            WorkflowStepConfig(
                type=WorkflowType.RUN,
                days=3,
            ),
            WorkflowStepConfig(
                type=WorkflowType.SAVE_CONTEXT,
                target_agent=AgentFilterConfig(
                    agent_class=(SocietyAgent,),
                ),
                key="attitude",
                save_as="guncontrol_attitude_final",
            ),
            WorkflowStepConfig(
                type=WorkflowType.SAVE_CONTEXT,
                target_agent=AgentFilterConfig(
                    agent_class=(SocietyAgent,),
                ),
                key="chat_histories",
                save_as="guncontrol_chat_histories",
            ),
        ],
        environment=EnvironmentConfig(
            start_tick=6 * 60 * 60,
        ),
    ),
)
config = default(config)


async def main():
    agentsociety = AgentSociety.create(config)
    try:
        await agentsociety.init()
        await agentsociety.run()
    finally:
        await agentsociety.close()


if __name__ == "__main__":
    asyncio.run(main())
