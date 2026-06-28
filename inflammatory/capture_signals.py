import asyncio, json, time
from agentsociety.cityagent import default, SocietyAgent
from agentsociety.configs import AgentsConfig, Config, EnvConfig, ExpConfig, LLMConfig, MapConfig
from agentsociety.configs.agent import AgentConfig
from agentsociety.configs.exp import WorkflowStepConfig, WorkflowType
from agentsociety.environment import EnvironmentConfig
from agentsociety.llm import LLMProviderType
from agentsociety.simulation import AgentSociety
from agentsociety.storage import DatabaseConfig

NUM_CITIZENS = 30
CAPTURE_STEPS = 25
OUT = "llm_signals.jsonl"
PROMPTS_OUT = "prompts_head.jsonl"
PREFIX_CHARS = 2000

_STEP = {"i": -1}
_prompts = []

config = Config(
    llm=[LLMConfig(provider=LLMProviderType.VLLM, base_url="http://localhost:8000/v1",
                   api_key="EMPTY", model="Qwen/Qwen2.5-7B-Instruct", concurrency=12)],
    env=EnvConfig(db=DatabaseConfig(enabled=True, db_type="sqlite", pg_dsn=None)),
    map=MapConfig(file_path="../map.pb"),
    agents=AgentsConfig(citizens=[AgentConfig(agent_class="citizen", number=NUM_CITIZENS)]),
    exp=ExpConfig(name="signal_capture",
                  workflow=[WorkflowStepConfig(type=WorkflowType.STEP, steps=1, ticks_per_step=300)],
                  environment=EnvironmentConfig(start_tick=6 * 60 * 60)),
    logging_level="WARNING",
)
config = default(config)


async def main():
    sim = AgentSociety.create(config)
    rows = []
    try:
        await sim.init()
        n_agents = len(await sim.filter())
        n_citizens = len(await sim.filter(types=(SocietyAgent,)))

        _orig_atext = sim.llm.atext_request
        async def _capturing_atext(dialog, *args, **kwargs):
            try:
                text = "\n".join(
                    (m.get("content", "") if isinstance(m, dict) else str(m)) or ""
                    for m in dialog
                )
            except Exception:
                text = str(dialog)
            _prompts.append({"step": _STEP["i"], "len_chars": len(text),
                             "head": text[:PREFIX_CHARS]})
            return await _orig_atext(dialog, *args, **kwargs)
        sim.llm.atext_request = _capturing_atext

        for step in range(CAPTURE_STEPS):
            _STEP["i"] = step
            t0 = time.perf_counter()
            for attempt in range(4):
                try:
                    logs = await sim.step(300)
                    break
                except Exception as e:
                    if attempt == 3:
                        raise
                    print(f"[retry] {type(e).__name__}; {attempt + 1}/3", flush=True)
                    await asyncio.sleep(3)
            wall = time.perf_counter() - t0
            calls = logs.llm_log
            for c in calls:
                rows.append({"step": step, "n_agents": n_agents, "n_citizens": n_citizens,
                             "in_tokens": c.get("input_tokens", 0),
                             "out_tokens": c.get("output_tokens", 0),
                             "latency_s": c.get("consumption", 0.0)})
            print(f"[step {step}] llm_calls={len(calls)} "
                  f"calls/citizen={len(calls) / n_citizens:.1f} wall={wall:.2f}s", flush=True)
            with open(OUT, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(PROMPTS_OUT, "w") as f:
                for p in _prompts:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
    finally:
        await sim.close()
    print(f"[done] {len(rows)} LLM calls over {CAPTURE_STEPS} steps -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
