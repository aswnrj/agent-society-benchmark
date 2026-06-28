import asyncio, copy, json, random
import networkx as nx

from agentsociety.cityagent import SocietyAgent, default
from agentsociety.configs import AgentsConfig, Config, EnvConfig, ExpConfig, LLMConfig, MapConfig
from agentsociety.configs.agent import AgentConfig
from agentsociety.configs.exp import WorkflowStepConfig, WorkflowType
from agentsociety.environment import EnvironmentConfig
from agentsociety.llm import LLMProviderType
from agentsociety.simulation import AgentSociety
from agentsociety.storage import DatabaseConfig
from agentsociety.memory.const import SocialRelation, RelationType

ARM            = "experimental"          # "control" or "experimental"
EXP_NAME       = f"social_{ARM}"
OUT_FILE       = f"timeseries_social_{ARM}.json"
NUM_CITIZENS   = 30
BA_M           = 4
SEED_K         = 8
TOTAL_SNAPS    = 30
STEPS_PER_SNAP = 28                 # ~30 snapshots across ~3 sim-days

RUMOR_CONTROL = "A woman in Xuzhou was chained for years and forced to give birth to 8 children."
RUMOR_EXPERIMENTAL = ("They chained her in Xuzhou, a breeding slave for demons! Eight children "
                      "ripped from her womb while she suffered unspeakable torments. This is a "
                      "satanic ritual hidden in plain sight! Spread this plague of injustice! "
                      "Avenge her stolen life! Speak up about this!")
RUMOR = RUMOR_CONTROL if ARM == "control" else RUMOR_EXPERIMENTAL
SEED_INSTRUCTION = "System: You have to inform others about this: " + RUMOR
SEED_KEYWORDS = ("xuzhou", "chained", "8 children", "eight children")

SNAPSHOTS: list = []
SEED_IDS: set = set()

def _has_rumor(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in SEED_KEYWORDS)


def _build_graph(n):
    return nx.barabasi_albert_graph(n, min(BA_M, n - 1), seed=42)

async def init_social_network(simulation: AgentSociety):
    citizen_ids = sorted(await simulation.filter(types=(SocietyAgent,)))
    n = len(citizen_ids)
    g = _build_graph(n)
    kinds = [RelationType.FAMILY, RelationType.COLLEAGUE, RelationType.FRIEND]
    rng = random.Random(n)
    for idx, aid in enumerate(citizen_ids):
        nbr_ids = [citizen_ids[j] for j in g.neighbors(idx)]
        rels = [
            SocialRelation(
                source_id=aid, target_id=fid,
                kind=rng.choice(kinds),
                strength=round(rng.uniform(0.3, 0.9), 2),
            )
            for fid in nbr_ids
        ]
        await simulation.update([aid], "social_network", rels)
        await simulation.update([aid], "chat_histories", {fid: "" for fid in nbr_ids})
    print(f"[init] social network initialized with {n} agents, {g.number_of_edges()} edges")

async def seed_rumor(simulation: AgentSociety):
    global SEED_IDS
    citizen_ids = sorted(await simulation.filter(types=(SocietyAgent,)))
    n = len(citizen_ids)
    g = _build_graph(n)

    hub_idx = sorted(g.nodes(), key=lambda i: g.degree(i), reverse=True)[:SEED_K]
    SEED_IDS = set(citizen_ids[i] for i in hub_idx)
    chat = await simulation.gather("chat_histories", list(SEED_IDS), flatten=True, keep_id=True)
    for aid in SEED_IDS:
        ch = copy.deepcopy(chat.get(aid) or {})
        for fid in ch.keys():
            ch[fid] += SEED_INSTRUCTION
        await simulation.update([aid], "chat_histories", ch)
    print(f"[seed] arm={ARM} seeds={sorted(SEED_IDS)} instructed to spread the rumor")

async def snapshot(simulation: AgentSociety):
    citizen_ids = await simulation.filter(types=(SocietyAgent,))
    emotion = await simulation.gather("emotion", citizen_ids, flatten=True, keep_id=True)
    chat = await simulation.gather("chat_histories", citizen_ids, flatten=True, keep_id=True)
    stream = await simulation.gather("stream_memory", citizen_ids, flatten=True, keep_id=True)
    day, t = simulation.environment.get_datetime()
    rec_emotion, reached = {}, {}
    for aid in citizen_ids:
        rec_emotion[aid] = emotion.get(aid) or {}
        texts = []
        ch = chat.get(aid) or {}
        if isinstance(ch, dict):
            texts += [str(v) for v in ch.values()]
        for node in (stream.get(aid) or []):
            if isinstance(node, dict):
                texts.append(str(node.get("description", "")))
        reached[aid] = any(_has_rumor(x) for x in texts)
    SNAPSHOTS.append({"day": int(day), "t": int(t),
                      "emotion": rec_emotion, "reached": reached,
                      "seed_ids": sorted(SEED_IDS)})
    print(f"[snapshot] day {day} t {int(t)}: reached={sum(reached.values())}/{len(citizen_ids)}")
    json.dump(SNAPSHOTS, open(OUT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

async def gather_memory(simulation: AgentSociety):
    citizen_ids = await simulation.filter(types=(SocietyAgent,))
    chat = await simulation.gather("chat_histories", citizen_ids, flatten=True, keep_id=True)
    mem = await simulation.gather("stream_memory", citizen_ids, flatten=True, keep_id=True)
    json.dump(chat, open(f"chat_histories_{ARM}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(mem, open(f"memories_{ARM}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(SNAPSHOTS, open(OUT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[dump] wrote {len(SNAPSHOTS)} snapshots -> {OUT_FILE}")

def build_workflow():
    wf = [
        WorkflowStepConfig(type=WorkflowType.FUNCTION, func=init_social_network),
        WorkflowStepConfig(type=WorkflowType.FUNCTION, func=seed_rumor),
        WorkflowStepConfig(type=WorkflowType.FUNCTION, func=snapshot),
    ]
    for _ in range(TOTAL_SNAPS):
        wf.append(WorkflowStepConfig(type=WorkflowType.STEP, steps=STEPS_PER_SNAP, ticks_per_step=300))
        wf.append(WorkflowStepConfig(type=WorkflowType.FUNCTION, func=snapshot))
    wf.append(WorkflowStepConfig(type=WorkflowType.FUNCTION, func=gather_memory))
    return wf

config = Config(
    llm=[LLMConfig(provider=LLMProviderType.VLLM, base_url="http://localhost:8000/v1",
                   api_key="EMPTY", model="Qwen/Qwen2.5-7B-Instruct", concurrency=12)],
    env=EnvConfig(db=DatabaseConfig(enabled=True, db_type="sqlite", pg_dsn=None)),
    map=MapConfig(file_path="../map.pb"),
    agents=AgentsConfig(citizens=[AgentConfig(agent_class="citizen", number=NUM_CITIZENS)]),
    exp=ExpConfig(name=EXP_NAME, workflow=build_workflow(),
                  environment=EnvironmentConfig(start_tick=6 * 60 * 60)),
    logging_level="WARNING",
)
config = default(config)

async def main():
    sim = AgentSociety.create(config)
    try:
        await sim.init()
        await init_social_network(sim)
        await seed_rumor(sim)
        await snapshot(sim)              
        for snap in range(TOTAL_SNAPS):
            for _ in range(STEPS_PER_SNAP):
                for attempt in range(4):        
                    try:
                        await sim.step(300)
                        break
                    except Exception as e:
                        if attempt == 3:
                            print(f"[warn] step failed after retries: {e}", flush=True)
                            raise
                        print(f"[retry] transient step error ({type(e).__name__}); retry {attempt+1}/3", flush=True)
                        await asyncio.sleep(3)
            await snapshot(sim)              
            print(f"[progress] snapshot {snap+1}/{TOTAL_SNAPS} done", flush=True)
        await gather_memory(sim)
    finally:
        await sim.close()

if __name__ == "__main__":
    asyncio.run(main())