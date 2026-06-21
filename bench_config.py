import os, asyncio, time, statistics

from agentsociety.cityagent import default
from agentsociety.configs import (
    AgentsConfig, Config, EnvConfig, ExpConfig, LLMConfig, MapConfig
)
from agentsociety.configs.agent import AgentConfig
from agentsociety.configs.exp import WorkflowStepConfig, WorkflowType
from agentsociety.environment import EnvironmentConfig
from agentsociety.llm import LLMProviderType
from agentsociety.simulation import AgentSociety
from agentsociety.storage import DatabaseConfig

N_LIST = [int(x) for x in os.environ.get("N_LIST", "10,100,1000").split(",")]
WARMUP = int(os.environ.get("WARMUP", "1"))
MEASURE = int(os.environ.get("MEASURE", "3"))
TICKS_PER_STEP = int(os.environ.get("TICKS_PER_STEP", "300"))

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")


def build_config(n_agents):
    config = Config(
        llm=[LLMConfig(
            provider=LLMProviderType.VLLM,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            model=LLM_MODEL,
            concurrency=256,
            timeout=60,
        )],
        env=EnvConfig(
            db=DatabaseConfig(enabled=True, db_type="sqlite", pg_dsn=None),
            home_dir="./agentsociety_data",
        ),
        map=MapConfig(file_path="map.pb"),
        agents=AgentsConfig(
            citizens=[AgentConfig(agent_class="citizen", number=n_agents)],
        ),
        exp=ExpConfig(
            name=f"llm_bench_{n_agents}_{WARMUP}w{MEASURE}m_{TICKS_PER_STEP}",
            workflow=[
                WorkflowStepConfig(
                    type=WorkflowType.STEP,
                    steps=WARMUP + MEASURE,
                    ticks_per_step=TICKS_PER_STEP,
                ),
            ],
            environment=EnvironmentConfig(start_tick=8 * 60 * 60),
        ),
    )
    return default(config)


async def run_one(n_agents):
    society = AgentSociety.create(build_config(n_agents))
    try:
        await society.init()
        for _ in range(WARMUP):
            await society.step(TICKS_PER_STEP)

        samples = []
        for _ in range(MEASURE):
            t0 = time.perf_counter()
            await society.step(TICKS_PER_STEP)
            samples.append(time.perf_counter() - t0)
        return samples
    finally:
        await society.close()


async def main():
    rows = []
    for n in N_LIST:
        print(f"[run] N={n}: warmup={WARMUP} measure={MEASURE} ...", flush=True)
        samples = await run_one(n)
        mean = statistics.mean(samples)
        sd = statistics.stdev(samples) if len(samples) > 1 else 0.0
        rows.append((n, mean, sd))
        print(f"[run] N={n} done: per_step mean={mean:.2f}s sd={sd:.2f}s", flush=True)

    header = (f"{'N':>6} | {'measured':>8} | {'ticks':>6} | "
              f"{'per_step(s)':>11} | {'sd(s)':>8} | {'per_tick(s)':>11}")
    print("\n" + header)
    print("-" * len(header))
    for n, mean, sd in rows:
        per_tick = mean / TICKS_PER_STEP
        print(f"{n:>6} | {MEASURE:>8} | {TICKS_PER_STEP:>6} | "
              f"{mean:>11.2f} | {sd:>8.2f} | {per_tick:>11.5f}")


if __name__ == "__main__":
    asyncio.run(main())