"""
Multi-Backend Prefix Cache Router

Features:
- Sticky routing by program_id (preserves prefix cache)
- Per-backend usage tracking and pause/resume
- Least-loaded backend assignment for new programs
"""

import asyncio
import hashlib
import json
import math
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Callable, Awaitable

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client.parser import text_string_to_metric_families

# Configuration
VLLM_BACKENDS = os.getenv(
    "VLLM_BACKENDS", 
    "http://localhost:8100"
).split(",")

PAUSE_TRIGGER_USAGE = 0.95
PAUSE_TARGET_USAGE = 0.90
PAUSE_COOLDOWN_S = 3
PAUSE_STEP = 0.1
RESUME_TRIGGER_USAGE = 0.85
RESUME_TARGET_USAGE = 0.90
RESUME_COOLDOWN_S = 12
RESUME_STEP = 0.05
TRANSFER_TRIGGER_USAGE = 0.50
TRANSFER_TARGET_USAGE = 0.70
TRANSFER_COOLDOWN_S = 12
TRANSFER_STEP = 0.05
KV_CACHE_TOKEN_BUDGET = 788976
OUTPUT_TOKEN_ESTIMATE = 500

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProgramState:
    context_len: int
    step_count: int = 0
    inflight: bool = False
    paused: bool = False
    pause_requested: bool = False
    transfer_target: Optional[str] = None
    resume_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def est_tokens(self) -> int:
        return self.context_len + OUTPUT_TOKEN_ESTIMATE

    def __post_init__(self):
        if not self.paused:
            self.resume_event.set()


@dataclass
class BackendState:
    url: str
    usage: float = 0.0
    healthy: bool = True
    running_requests: float = 0.0
    waiting_requests: float = 0.0
    programs: Dict[str, ProgramState] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_resume_time: float = 0.0
    last_pause_time: float = 0.0
    last_transfer_time: float = 0.0

    @property
    def metrics_url(self) -> str:
        return f"{self.url}/metrics"

    @property
    def completions_url(self) -> str:
        return f"{self.url}/v1/chat/completions"

    @property
    def total_inflight(self) -> int:
        return sum(1 for p in self.programs.values() if p.inflight)

    @property
    def paused_count(self) -> int:
        return sum(1 for p in self.programs.values() if p.paused)


class MultiBackendRouter:
    def __init__(self, backend_urls: List[str]) -> None:
        self.backends: Dict[str, BackendState] = {
            url: BackendState(url=url) for url in backend_urls
        }
        self.program_affinity: Dict[str, str] = {}  # program_id -> backend_url
        self._transfer_rr_index = 0
        
        self.client = httpx.AsyncClient(
            timeout=900.0,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        self.metrics_client = httpx.AsyncClient(timeout=5.0)
        self.monitor_task: Optional[asyncio.Task] = None

    async def start(self):
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Started router with {len(self.backends)} backends: {list(self.backends.keys())}")

    async def stop(self):
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        await self.client.aclose()
        await self.metrics_client.aclose()

    # -------------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_metric_value(line: str) -> Optional[float]:
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            value = float(parts[1])
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        return value

    def _extract_metrics_fallback(self, text: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
        usage = None
        running = None
        waiting = None
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            if line.startswith("vllm:kv_cache_usage_perc"):
                if usage is None:
                    usage = self._parse_metric_value(line)
            elif line.startswith("vllm:num_requests_running"):
                if running is None:
                    running = self._parse_metric_value(line)
            elif line.startswith("vllm:num_requests_waiting"):
                if waiting is None:
                    waiting = self._parse_metric_value(line)
            if usage is not None and running is not None and waiting is not None:
                break
        return usage, running, waiting

    async def _monitor_loop(self):
        while True:
            # Fetch usage from all backends in parallel
            tasks = [
                self._fetch_backend_usage(backend)
                for backend in self.backends.values()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Update pause state for each backend independently
            for backend in self.backends.values():
                async with backend.lock:
                    self._update_backend_pause_state(backend)

            await asyncio.sleep(0.5)

    async def _fetch_backend_usage(self, backend: BackendState):
        try:
            resp = await self.metrics_client.get(backend.metrics_url, timeout=3.0)
            resp.raise_for_status()

            running_req = None
            waiting_req = None
            usage_val = None

            try:
                families = list(text_string_to_metric_families(resp.text))
            except Exception as exc:
                logger.warning(
                    f"Failed to parse metrics from {backend.url}, falling back to line parsing: {exc!r}"
                )
                usage_val, running_req, waiting_req = self._extract_metrics_fallback(resp.text)
            else:
                for family in families:
                    if family.name == "vllm:kv_cache_usage_perc":
                        values = [
                            float(sample.value)
                            for sample in family.samples
                            if sample.name == family.name and math.isfinite(float(sample.value))
                        ]
                        if values:
                            usage_val = max(values)
                    elif family.name == "vllm:num_requests_running":
                        running_req = sum(
                            float(sample.value)
                            for sample in family.samples
                            if sample.name == family.name and math.isfinite(float(sample.value))
                        )
                    elif family.name == "vllm:num_requests_waiting":
                        waiting_req = sum(
                            float(sample.value)
                            for sample in family.samples
                            if sample.name == family.name and math.isfinite(float(sample.value))
                        )

            if usage_val is not None:
                backend.usage = usage_val / 100.0 if usage_val > 1.0 else usage_val
            if running_req is not None:
                backend.running_requests = running_req
            if waiting_req is not None:
                backend.waiting_requests = waiting_req
            backend.healthy = True
        except Exception as exc:
            logger.warning(f"Failed to fetch metrics from {backend.url}: {exc}")
            backend.healthy = False

    # -------------------------------------------------------------------------
    # Pause/Resume Logic (per-backend)
    # -------------------------------------------------------------------------

    def _update_backend_pause_state(self, backend: BackendState):
        now = time.time()
        usage = backend.usage

        if usage >= PAUSE_TRIGGER_USAGE:
            if now - backend.last_pause_time < PAUSE_COOLDOWN_S:
                return
            backend.last_pause_time = now
            tokens_to_pause = (
                max(0.0, usage - PAUSE_TARGET_USAGE)
                * KV_CACHE_TOKEN_BUDGET
                * PAUSE_STEP
            )
            transfer_tokens = self._schedule_transfers_on_backend(backend, now)
            tokens_to_pause = max(0.0, tokens_to_pause - transfer_tokens)
            self._pause_programs_on_backend(backend, tokens_to_pause)
        elif usage < RESUME_TRIGGER_USAGE:
            if now - backend.last_resume_time < RESUME_COOLDOWN_S:
                return
            backend.last_resume_time = now
            self._resume_programs_on_backend(backend, usage)

    def _pick_transfer_target(self, source_backend: BackendState) -> Optional[BackendState]:
        eligible = [
            b for b in self.backends.values()
            if b.url != source_backend.url
            and b.healthy
            and b.paused_count == 0
            and b.usage < TRANSFER_TRIGGER_USAGE
        ]
        if not eligible:
            return None
        idx = self._transfer_rr_index % len(eligible)
        self._transfer_rr_index += 1
        return eligible[idx]

    def _schedule_transfers_on_backend(self, backend: BackendState, now: float) -> float:
        if now - backend.last_transfer_time < TRANSFER_COOLDOWN_S:
            return 0.0
        target = self._pick_transfer_target(backend)
        if target is None:
            return 0.0
        transfer_budget = (
            max(0.0, TRANSFER_TARGET_USAGE - target.usage)
            * KV_CACHE_TOKEN_BUDGET
            * TRANSFER_STEP
        )
        if transfer_budget <= 0:
            return 0.0

        active = [
            (pid, state)
            for pid, state in backend.programs.items()
            if not state.paused
            and not state.pause_requested
            and state.transfer_target is None
        ]
        if not active:
            return 0.0

        active.sort(key=lambda item: item[1].context_len)
        transfer_tokens = 0.0
        transfer_pids = []

        for pid, state in active:
            if transfer_tokens >= transfer_budget:
                break
            state.transfer_target = target.url
            transfer_tokens += state.est_tokens
            transfer_pids.append(pid)

        if transfer_pids:
            backend.last_transfer_time = now
            logger.info(
                f"[{backend.url}] Scheduled transfer of {len(transfer_pids)} programs "
                f"to {target.url} (budget={transfer_budget}, tokens={transfer_tokens}): {transfer_pids}"
            )

        return transfer_tokens

    def _pause_programs_on_backend(
        self,
        backend: BackendState,
        tokens_to_pause: float,
    ):
        """Pause low-priority programs on this backend."""
        active = [
            (pid, state)
            for pid, state in backend.programs.items()
            if not state.paused
            and not state.pause_requested
            and state.transfer_target is None
        ]
        if not active:
            return

        if tokens_to_pause <= 0:
            return
        active.sort(key=lambda item: (-item[1].step_count, item[1].est_tokens))

        paused_tokens = 0
        newly_paused = []

        for i in range(len(active) - 1, -1, -1):
            if paused_tokens >= tokens_to_pause:
                break

            pid, state = active[i]
            if state.inflight:
                state.pause_requested = True
            else:
                state.paused = True
                state.resume_event.clear()

            paused_tokens += state.est_tokens
            newly_paused.append(pid)

        if newly_paused:
            logger.info(
                f"[{backend.url}] Paused {len(newly_paused)} programs "
                f"(usage={backend.usage:.2%}, tokens={paused_tokens}): {newly_paused}"
            )

    def _resume_programs_on_backend(self, backend: BackendState, usage: float):
        """Resume paused programs on this backend."""
        max_resume_tokens = (
            max(0.0, RESUME_TARGET_USAGE - usage)
            * KV_CACHE_TOKEN_BUDGET
            * RESUME_STEP
        )
        if max_resume_tokens <= 0:
            return

        paused = [
            (pid, state)
            for pid, state in backend.programs.items()
            if state.paused
        ]
        if not paused:
            return

        paused.sort(key=lambda item: (-item[1].step_count, item[1].est_tokens))

        resumed_tokens = 0
        resumed_pids = []

        for pid, state in paused:
            if resumed_tokens >= max_resume_tokens:
                break
            state.paused = False
            state.pause_requested = False
            state.resume_event.set()
            resumed_tokens += state.est_tokens
            resumed_pids.append(pid)

        if resumed_pids:
            logger.info(
                f"[{backend.url}] Resumed {len(resumed_pids)} programs "
                f"(usage={usage:.2%}, budget={max_resume_tokens}): {resumed_pids}"
            )

    # -------------------------------------------------------------------------
    # Backend Selection (Sticky + Load Balancing)
    # -------------------------------------------------------------------------

    def _get_backend_for_program(self, program_id: str) -> BackendState:
        """Get backend for a program, using sticky routing or least-loaded assignment."""
        # Check existing affinity
        if program_id in self.program_affinity:
            backend_url = self.program_affinity[program_id]
            backend = self.backends.get(backend_url)
            if backend and backend.healthy:
                return backend
            # Backend is unhealthy, need to reassign
            logger.warning(
                f"Backend {backend_url} unhealthy for {program_id}, reassigning"
            )
            del self.program_affinity[program_id]

        # New program: assign to least loaded backend
        if program_id == "default":
            logger.warning("Missing job_id in extra_body; routing will not be balanced across backends.")
        backend = self._pick_least_loaded_backend(program_id)
        self.program_affinity[program_id] = backend.url
        logger.debug(f"Assigned {program_id} to {backend.url}")
        return backend

    def _pick_least_loaded_backend(self, program_id: str) -> BackendState:
        """Select a healthy backend, prefer those with no paused programs."""
        healthy = [b for b in self.backends.values() if b.healthy]
        if not healthy:
            # Fallback: use any backend
            logger.warning("No healthy backends, using first available")
            return list(self.backends.values())[0]

        candidates = [b for b in healthy if b.paused_count == 0] or healthy

        def score(b: BackendState) -> float:
            return b.running_requests + b.waiting_requests * 4.0

        scored = [(b, score(b)) for b in candidates]
        min_score = min(val for _, val in scored)
        best = [b for b, val in scored if val == min_score]
        if len(best) == 1:
            return best[0]
        digest = hashlib.sha256(program_id.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:8], "big") % len(best)
        return best[idx]

    async def _apply_pending_transfer(
        self,
        program_id: str,
        backend: BackendState,
        is_last_step: bool,
    ) -> BackendState:
        async with backend.lock:
            state = backend.programs.get(program_id)
            if state is None or state.transfer_target is None:
                return backend
            if is_last_step:
                state.transfer_target = None
                return backend
            target_url = state.transfer_target

        target_backend = self.backends.get(target_url)
        if target_backend is None or not target_backend.healthy:
            async with backend.lock:
                state = backend.programs.get(program_id)
                if state and state.transfer_target == target_url:
                    state.transfer_target = None
            return backend

        if target_backend.url == backend.url:
            async with backend.lock:
                state = backend.programs.get(program_id)
                if state:
                    state.transfer_target = None
            return backend

        first, second = sorted([backend, target_backend], key=lambda b: b.url)
        async with first.lock:
            async with second.lock:
                state = backend.programs.get(program_id)
                if (
                    state is None
                    or state.transfer_target != target_url
                    or state.inflight
                    or state.paused
                ):
                    return backend
                del backend.programs[program_id]
                target_backend.programs[program_id] = state
                self.program_affinity[program_id] = target_url
                state.transfer_target = None
                state.resume_event.set()
                return target_backend

    # -------------------------------------------------------------------------
    # Request Proxying
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_total_tokens(payload: Any) -> Optional[int]:
        if not isinstance(payload, dict):
            return None
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        if "total_tokens" in usage:
            val = usage.get("total_tokens")
            if isinstance(val, (int, float)) and math.isfinite(val):
                return int(val)
        return None

    @staticmethod
    def _filtered_headers(headers: httpx.Headers) -> Dict[str, str]:
        hop_by_hop = {"content-length", "transfer-encoding", "connection"}
        return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}

    async def proxy_request(
        self,
        backend: BackendState,
        payload: Dict[str, Any],
        *,
        on_total_tokens: Callable[[int], Awaitable[None]] | None = None,
    ) -> Response:
        url = backend.completions_url

        if payload.get("stream"):
            stream_options = payload.get("stream_options")
            if stream_options is None:
                payload["stream_options"] = {"include_usage": True}
            elif isinstance(stream_options, dict):
                stream_options.setdefault("include_usage", True)

            resp_cm = self.client.stream("POST", url, json=payload)
            resp = await resp_cm.__aenter__()
            headers = self._filtered_headers(resp.headers)
            status = resp.status_code
            media_type = resp.headers.get("content-type")

            async def iterator():
                buffer = b""
                total_tokens: Optional[int] = None
                try:
                    async for chunk in resp.aiter_raw():
                        buffer += chunk
                        while b"\n\n" in buffer:
                            event, buffer = buffer.split(b"\n\n", 1)
                            for line in event.split(b"\n"):
                                if not line.startswith(b"data:"):
                                    continue
                                data = line[5:].strip()
                                if not data or data == b"[DONE]":
                                    continue
                                if total_tokens is not None:
                                    continue
                                try:
                                    payload_obj = json.loads(data)
                                except Exception:
                                    continue
                                extracted = self._extract_total_tokens(payload_obj)
                                if extracted is not None:
                                    total_tokens = extracted
                        yield chunk
                finally:
                    await resp_cm.__aexit__(None, None, None)
                    if total_tokens is not None and on_total_tokens is not None:
                        await on_total_tokens(total_tokens)

            return StreamingResponse(
                iterator(),
                status_code=status,
                headers=headers,
                media_type=media_type,
            )

        resp = await self.client.post(url, json=payload)
        total_tokens: Optional[int] = None
        try:
            payload_obj = resp.json()
        except Exception:
            payload_obj = None
        extracted = self._extract_total_tokens(payload_obj)
        if extracted is not None:
            total_tokens = extracted
        if total_tokens is not None and on_total_tokens is not None:
            await on_total_tokens(total_tokens)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=self._filtered_headers(resp.headers),
            media_type=resp.headers.get("content-type"),
        )


# =============================================================================
# FastAPI Application
# =============================================================================

router = MultiBackendRouter(VLLM_BACKENDS)
app = FastAPI(title="Multi-Backend Prefix Cache Router")


@app.on_event("startup")
async def startup_event():
    await router.start()


@app.on_event("shutdown")
async def shutdown_event():
    await router.stop()


def _get_program_id(payload: Dict[str, Any], _request: Request) -> str:
    if "job_id" in payload:
        return str(payload["job_id"])
    extra_body = payload.get("extra_body", {})
    if isinstance(extra_body, dict) and "job_id" in extra_body:
        return str(extra_body["job_id"])
    return "default"


@app.post("/v1/chat/completions")
async def route_chat_completions(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    program_id = _get_program_id(payload, request)
    extra_body = payload.get("extra_body", {})
    if isinstance(extra_body, dict) and "is_last_step" in extra_body:
        is_last_step = extra_body.get("is_last_step", False)
    else:
        is_last_step = payload.get("is_last_step", False)

    # Get the backend for this program (sticky routing)
    backend = router._get_backend_for_program(program_id)
    backend = await router._apply_pending_transfer(
        program_id, backend, is_last_step
    )
    while True:
        async with backend.lock:
            if program_id not in backend.programs:
                backend.programs[program_id] = ProgramState(
                    context_len=0, step_count=0
                )
            state = backend.programs[program_id]

            if is_last_step and state.paused:
                state.paused = False
                state.pause_requested = False
                state.resume_event.set()

            if not state.paused:
                state.inflight = True
                state.step_count += 1
                break

            wait_event = state.resume_event

        await wait_event.wait()

    async def update_total_tokens(tokens: int) -> None:
        async with backend.lock:
            state = backend.programs.get(program_id)
            if state is not None:
                state.context_len = tokens

    try:
        return await router.proxy_request(backend, payload, on_total_tokens=update_total_tokens)
    finally:
        async with backend.lock:
            if program_id in backend.programs:
                state = backend.programs[program_id]
                state.inflight = False

                if is_last_step:
                    del backend.programs[program_id]
                    if program_id in router.program_affinity:
                        del router.program_affinity[program_id]
                    logger.info(f"[{backend.url}] Program {program_id} completed")
                elif state.pause_requested and not state.inflight:
                    state.paused = True
                    state.resume_event.clear()


@app.get("/programs")
async def list_programs():
    """List all programs across all backends."""
    result = {}
    for backend in router.backends.values():
        async with backend.lock:
            for pid, s in backend.programs.items():
                result[pid] = {
                    "backend": backend.url,
                    "context_len": s.context_len,
                    "step": s.step_count,
                    "inflight": s.inflight,
                    "paused": s.paused,
                    "pause_requested": s.pause_requested,
                }
    return JSONResponse(result)


@app.get("/backends")
async def list_backends():
    """List all backends and their status."""
    result = {}
    for url, backend in router.backends.items():
        async with backend.lock:
            result[url] = {
                "healthy": backend.healthy,
                "usage": backend.usage,
                "program_count": len(backend.programs),
                "inflight": backend.total_inflight,
                "paused_count": backend.paused_count,
            }
    return JSONResponse(result)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    healthy_count = sum(1 for b in router.backends.values() if b.healthy)
    return JSONResponse({
        "status": "healthy" if healthy_count > 0 else "degraded",
        "healthy_backends": healthy_count,
        "total_backends": len(router.backends),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300, log_level="info")
