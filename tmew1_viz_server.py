"""
Real-time visualization server for TMEW-1 training.

Runs a WebSocket server in a background thread.  The training loop calls
send_metrics() / send_episode() and the browser dashboard receives live updates.

Protocol (JSON over WebSocket):
  {type: "metrics", ...}      — training metric snapshot
  {type: "episode_start", ...} — new episode replay beginning
  {type: "frame", ...}        — single timestep within a replay
  {type: "query", ...}        — query result for the replayed episode
"""

from __future__ import annotations

import asyncio
import json
import random
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# WebSocket broadcast machinery
# ---------------------------------------------------------------------------

_clients: Set[Any] = set()
_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_broadcast_queue: asyncio.Queue = None  # type: ignore[assignment]


def _serialise(obj: Any) -> str:
    """JSON-safe serialisation (handles numpy scalars etc.)."""
    def default(o: Any) -> Any:
        try:
            return float(o)
        except (TypeError, ValueError):
            return str(o)
    return json.dumps(obj, default=default)


async def _handler(ws: Any) -> None:
    _clients.add(ws)
    try:
        async for _ in ws:
            pass  # we only broadcast, ignore incoming
    finally:
        _clients.discard(ws)


async def _broadcaster() -> None:
    while True:
        msg = await _broadcast_queue.get()
        if _clients:
            payload = _serialise(msg)
            dead: List[Any] = []
            for ws in _clients:
                try:
                    await ws.send(payload)
                except Exception:
                    dead.append(ws)
            for d in dead:
                _clients.discard(d)


async def _run_server(host: str, port: int) -> None:
    global _broadcast_queue
    _broadcast_queue = asyncio.Queue()
    asyncio.ensure_future(_broadcaster())
    try:
        import websockets  # type: ignore
        async with websockets.serve(_handler, host, port):
            print(f"[viz] WebSocket server running on ws://{host}:{port}")
            await asyncio.Future()  # run forever
    except ImportError:
        print("[viz] 'websockets' package not installed — run: pip install websockets")
        return


def start_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    """Start the viz WebSocket server in a daemon thread.  Safe to call once."""
    global _loop, _thread
    if _thread is not None and _thread.is_alive():
        return

    def _run() -> None:
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        _loop.run_until_complete(_run_server(host, port))

    _thread = threading.Thread(target=_run, daemon=True, name="viz-server")
    _thread.start()


def _enqueue(msg: Dict[str, Any]) -> None:
    """Thread-safe enqueue for broadcast."""
    if _loop is not None and _broadcast_queue is not None:
        _loop.call_soon_threadsafe(_broadcast_queue.put_nowait, msg)


# ---------------------------------------------------------------------------
# Public API — called from the training loop
# ---------------------------------------------------------------------------

def send_metrics(
    *,
    step: int,
    tier: int,
    epoch: int,
    metrics: Dict[str, float],
    pnn_states: str = "",
) -> None:
    """Broadcast a training metrics snapshot."""
    _enqueue({
        "type": "metrics",
        "step": step,
        "tier": tier,
        "epoch": epoch,
        "pnn": pnn_states,
        **{k: round(float(v), 6) for k, v in metrics.items()},
    })


def send_episode_replay(
    *,
    world_cfg: Any,
    tier: Any,
    seed: int = -1,
    model: Any = None,
    query_head: Any = None,
    device: str = "cpu",
) -> None:
    """Generate one episode and broadcast it frame-by-frame for 3-D playback.

    If *model* and *query_head* are provided the model's query predictions are
    included alongside ground-truth answers.
    """
    # Lazy import to avoid circular deps
    from tmew1_train import (
        WorldConfig, CurriculumTier, WorldState, _make_world, _step_world,
        generate_episode,
    )
    from tmew1_queries import (
        QUERY_TYPES, generate_episode_with_queries,
    )
    import torch, numpy as np

    if seed < 0:
        seed = random.randint(0, 999_999)

    cfg: WorldConfig = world_cfg
    template = random.Random(seed).choice(tier.template_pool)
    state = _make_world(cfg, template, tier.max_delay, tier.occlusion, seed)
    T = tier.max_episode_length

    # Episode-start message
    _enqueue({
        "type": "episode_start",
        "template": template,
        "tier": tier.tier,
        "T": T,
        "grid_w": cfg.grid_w,
        "grid_h": cfg.grid_h,
        "num_entities": len(state.entities),
        "latent_rule": state.active_rule,
        "has_occluder": state.occluder is not None,
        "occluder": list(state.occluder) if state.occluder else None,
    })

    # Track token holder for handoff episodes
    holder_id: Optional[int] = None
    if template in ("handoff", "false_cue"):
        holder_id = 0

    handoff_count = 0

    for t in range(T):
        events = _step_world(state, cfg, t, template, tier.max_delay)

        # Handoff tracking (mirrors _step_world_with_handoff)
        ev_handoff = False
        new_holder = -1
        if holder_id is not None:
            for i in range(len(state.entities)):
                for j in range(i + 1, len(state.entities)):
                    a, b = state.entities[i], state.entities[j]
                    if abs(a.x - b.x) + abs(a.y - b.y) <= 1:
                        if a.id == holder_id:
                            holder_id = b.id
                            ev_handoff = True
                            new_holder = b.id
                            handoff_count += 1
                            break
                        elif b.id == holder_id:
                            holder_id = a.id
                            ev_handoff = True
                            new_holder = a.id
                            handoff_count += 1
                            break
                if ev_handoff:
                    break

        entities_data = []
        for e in state.entities:
            entities_data.append({
                "id": e.id,
                "x": e.x,
                "y": e.y,
                "color": e.color,
                "tagged": e.tagged,
                "visible": e.visible,
                "holds_token": (holder_id == e.id) if holder_id is not None else False,
            })

        _enqueue({
            "type": "frame",
            "t": t,
            "entities": entities_data,
            "trigger": events.get("trigger", False),
            "alarm_fire": events.get("alarm_fire", False),
            "alarm_countdown": max(0, state.alarm_in),
            "chain2_trigger": events.get("chain2_trigger", False),
            "chain2_fire": events.get("chain2_fire", False),
            "chain2_countdown": max(0, state.chain2_alarm_in),
            "handoff": ev_handoff,
            "new_holder": new_holder,
            "holder_id": holder_id if holder_id is not None else -1,
            "occluded_ids": events.get("occluded_ids", []),
            "handoff_count": handoff_count,
        })

    # Generate queries with ground-truth answers
    ep = generate_episode_with_queries(cfg, tier, num_queries=len(QUERY_TYPES), seed=seed)
    for q in ep.queries:
        query_msg: Dict[str, Any] = {
            "type": "query",
            "qtype": q.qtype,
            "target": q.target,
            "time_asked": q.time_asked,
            "is_binary": q.is_binary,
            "prediction": None,
            "correct": None,
        }

        # If model is available, get its prediction
        if model is not None and query_head is not None:
            try:
                import torch
                model.eval()
                query_head.eval()
                with torch.no_grad():
                    vis = torch.from_numpy(ep.vision).unsqueeze(0).to(device)[:, :-1]
                    aud = torch.from_numpy(ep.audio).unsqueeze(0).to(device)[:, :-1]
                    num = torch.from_numpy(ep.numeric).unsqueeze(0).to(device)[:, :-1]
                    txt = torch.from_numpy(ep.text).squeeze(-1).unsqueeze(0).to(device)[:, :-1]
                    output = model(
                        vision=vis if "vision" in tier.enabled_modalities else None,
                        audio=aud if "audio" in tier.enabled_modalities else None,
                        numeric=num if "numeric" in tier.enabled_modalities else None,
                        text_tokens=txt if "text" in tier.enabled_modalities else None,
                    )
                    from tmew1_queries import QUERY_TYPE_TO_IDX
                    qt = torch.tensor([[QUERY_TYPE_TO_IDX[q.qtype]]]).to(device)
                    qtime = torch.tensor([[min(q.time_asked, output.sequence.size(1) - 1)]]).to(device)

                    from tmew1_run import augment_sequence_with_holder_audio
                    aug_seq = augment_sequence_with_holder_audio(
                        output.sequence,
                        torch.from_numpy(ep.audio).unsqueeze(0).to(device),
                        max_entities=cfg.max_entities,
                        use_audio="audio" in tier.enabled_modalities,
                    )
                    e_logits, b_logits = query_head(aug_seq, qtime, qt)
                    if q.is_binary:
                        pred = int(b_logits[0, 0].argmax().item())
                    else:
                        pred = int(e_logits[0, 0].argmax().item())
                    query_msg["prediction"] = pred
                    query_msg["correct"] = (pred == q.target)
                model.train()
                query_head.train()
            except Exception:
                pass  # silently skip if model inference fails

        _enqueue(query_msg)

    # Signal end of episode
    _enqueue({"type": "episode_end"})
