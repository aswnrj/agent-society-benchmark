import os, csv, asyncio, time, random, statistics

from agentsociety.environment.environment import (
    EnvironmentStarter, EnvironmentConfig,
)
from agentsociety.environment.mapdata import MapConfig
from agentsociety.environment.sim.person_service import PersonService
from agentsociety.s3 import S3Config
from pycityproto.city.trip.v2.trip_pb2 import TripMode


def make_person(home_aoi: int, targets: list[int], start_tick: int) -> dict:
    p = PersonService.default_person(return_dict=True)
    p["home"] = {"aoi_position": {"aoi_id": home_aoi}}
    p["schedules"] = [
        {
            "trips": [{
                "mode": TripMode.TRIP_MODE_DRIVE_ONLY,
                "end": {"aoi_position": {"aoi_id": aoi}},
                "departure_time": float(start_tick + 60 * i),
            }],
            "loop_count": 1,
            "departure_time": float(start_tick + 60 * i),
        }
        for i, aoi in enumerate(targets)
    ]
    return p

async def bench_for_n(env, aoi_ids, start_tick, n_agents, warmup=5, measured=50):
    # 1. Populate n agents
    for _ in range(n_agents):
        home = random.choice(aoi_ids)
        targets = [random.choice(aoi_ids) for _ in range(3)]
        await env.add_person(make_person(home, targets, start_tick))

    # 2. Warmup
    for _ in range(warmup):
        await env.step(1)

    # 3. Time individual steps
    samples = []
    for _ in range(measured):
        t0 = time.perf_counter()
        await env.step(1)
        samples.append(time.perf_counter() - t0)

    return statistics.mean(samples), statistics.stdev(samples)

async def main():
    os.makedirs("./.agentsociety", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    map_config = MapConfig(file_path="map.pb")
    env_config = EnvironmentConfig()
    s3 = S3Config(enabled=False)

    env = EnvironmentStarter(
        map_config=map_config,
        environment_config=env_config,
        s3config=s3,
        log_dir="./logs",
        home_dir="./.agentsociety",
    )
    await env.init()
    drive_aoi_ids = [
        a["id"] for a in env.map.get_all_aois() if a.get("driving_positions")
    ]
    if not drive_aoi_ids:
        raise RuntimeError("No AOIs with driving positions found in the map.")
    start_tick = env_config.start_tick

    results = []
    print(f"{'# agents':>10} | {'mean (s)':>12} | {'sd (s)':>12}")
    for n in (10**3, 10**4, 10**5):
        mean, sd = await bench_for_n(env, drive_aoi_ids, start_tick, n)
        results.append((n, mean, sd))
        print(f"{n:10} | {mean:12.3e} | {sd:12.3e}")

    with open("table2_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_agents", "mean_step_s", "sd_step_s"])
        w.writerows(results)
        
    await env.close()

if __name__ == "__main__":
    asyncio.run(main())