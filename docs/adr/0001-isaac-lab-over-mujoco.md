# ADR-0001: Isaac Lab over MuJoCo for Simulation

**Status:** Accepted
**Date:** 2026-05-06
**Deciders:** Project team

---

## Context

This project needs a physics simulator capable of:

1. Generating large quantities of synthetic manipulation demonstrations for an SO-101 robot arm
   running on an RTX 3080 (10 GB VRAM).
2. Supporting domain randomisation (DR) at scale — texture, lighting, joint damping, payload mass —
   without custom shim code.
3. Rendering image observations suitable for vision-based world models (DreamerV3, LeWorldModel).
4. Exporting robot assets in a format compatible with the broader robotics ecosystem.

The prior plan (`2026-05-03`) targeted MuJoCo + dm_control. After re-evaluation, Isaac Lab
was selected as the sole simulation layer.

---

## Decision

Use **NVIDIA Isaac Lab** (formerly Isaac Gym / OmniIsaac) as the simulator. MuJoCo is removed
from all packages except as an optional comparison target behind a soft import guard.

Isaac Lab is installed as a post-pixi manual step (`pixi run install-isaac-lab`) because it
requires the Isaac Sim runtime (~30 GB) and a CUDA-capable GPU.

---

## Rationale

### GPU parallelism

Isaac Lab can run thousands of parallel environments on a single GPU. For DR data generation,
this translates to orders-of-magnitude higher throughput than MuJoCo's CPU-bound parallel envs.
On the RTX 3080, 64–256 parallel environments are practical without hitting OOM.

### Native domain randomisation

Isaac Lab's `RandomizationManager` (later `EventManager`) provides first-class DR over:
- Physics parameters: joint friction, damping, payload mass
- Visual parameters: texture maps, albedo, specular, environment lighting
- Kinematics: link length perturbations

MuJoCo requires bespoke XML mutation or third-party wrappers (dm_control randomizers) to achieve
the same coverage. The Isaac Lab DR config maps directly onto YAML entries in
`packages/lerobot-isaac-configs/configs/`.

### USD ecosystem

Isaac Lab uses Universal Scene Description (USD) as the asset format. NVIDIA, Boston Dynamics,
and most major robotics vendors are converging on USD. Having the SO-101 as a USD asset:
- Enables import into Omniverse for human visualization
- Allows future use with Isaac Sim ROS2 bridge
- Provides a durable format independent of sim version

MuJoCo's MJCF format has no comparable ecosystem momentum.

### RTX 3080 fit

Isaac Lab headless rendering (EGL) allows ray-traced image observations at 64×64 without a
display. DreamerV3 and LeWorldModel both work at this resolution. The `--num_envs 8` default
in the training config is calibrated for 10 GB VRAM; it can be tuned down if needed.

---

## Consequences

**Positive:**
- Massively higher DR data throughput
- First-class USD asset pipeline
- Native Isaac Lab SO-101 URDF-to-USD conversion tooling available

**Negative:**
- Isaac Lab install is large (~30 GB) and requires NVIDIA drivers + CUDA
- CI cannot run Isaac-Lab-dependent tests without a GPU runner (mitigated by soft-import
  discipline and `@pytest.mark.requires_isaaclab` skip)
- MuJoCo expertise is not transferable; team must learn Isaac Lab's MDP API

**Neutral:**
- `pixi install` remains fast because Isaac Lab is a manual post-install step
- Existing MuJoCo-based data from the prior plan can still be used via the adapter layer

---

## Alternatives Considered

| Alternative | Why rejected |
|-------------|-------------|
| MuJoCo + dm_control | CPU-only parallelism too slow for DR at scale; MJCF ecosystem stagnating |
| PyBullet | Outdated renderer; no native DR; community largely moved on |
| Gazebo / gz-sim | ROS2-native but CPU-based physics; rendering not suitable for world models |
| Genesis | Promising but immature; USD support unproven; too early for production use |
