# Polarization experiment (gun control)

Reproduction of the opinion-polarization experiment from AgentSociety
(arXiv:2502.08691, Section 7.2 / Figure 16). 100 citizens start with an attitude
toward "Whether to support stronger gun control?" (0 = oppose, 10 = support) and
their opinions evolve over 3 simulated days.

Three arms:

- `control.py` — citizens only, no persuasion.
- `echo_chamber.py` — agents hear persuasive messages that match their own side.
- `back_firing.py` — agents hear persuasive messages from the opposing side.

The citizen attitudes and the agree/disagree agents' friend lists in `profiles/`
are the exact data shipped with the paper.

## Model

All results here were produced with **Qwen2.5-14B-Instruct** served locally via
vLLM. The paper used DeepSeek-V3, so absolute numbers differ; to match the paper
more closely, change the `LLMConfig` in each arm to a stronger model.

## Run

```bash
pip install -r requirements.txt
# start vLLM: vllm serve Qwen/Qwen2.5-14B-Instruct --port 8000
cd polarization
python control.py          # then echo_chamber.py, back_firing.py
python analyze.py --csv control.csv
```

`analyze.py` reports each citizen's opinion change in the four Figure 16
categories (More Polarized, Less Polarized, Opposed, Unchanged).

## Control result (Qwen2.5-14B-Instruct)

| Category | Ours | Paper |
|---|---:|---:|
| More Polarized | 56% | 39% |
| Less Polarized | 10% | 33% |
| Opposed | 34% | 10% |
| Unchanged | 0% | 18% |

See `results.txt` for the full summary and `control_artifacts.json` for the raw run.
