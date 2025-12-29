"""
Multi-Backend Prefix Cache Router

Features:
- Sticky routing by program_id (preserves prefix cache)
- Per-backend usage tracking and pause/resume
- Least-loaded backend assignment for new programs
"""

import asyncio
import math
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Configuration
VLLM_BACKENDS = os.getenv(
    "VLLM_BACKENDS", 
    "http://localhost:8100"
).split(",")

PAUSE_AT_USAGE = 0.95
RESUME_AT_USAGE = 0.90
KV_CACHE_TOKEN_BUDGET = 788976
OUTPUT_TOKEN_ESTIMATE = 500
LOW_USAGE_RELEASE = 0.5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_TOKENIZER_CACHE: Dict[str, PreTrainedTokenizerBase] = {}


@dataclass
class ProgramState:
    context_len: int
    step_count: int = 0
    inflight: int = 0
    paused: bool = False
    pause_requested: bool = False
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
    high_usage_hits: int = 0

    @property
    def metrics_url(self) -> str:
        return f"{self.url}/metrics"

    @property
    def completions_url(self) -> str:
        return f"{self.url}/v1/chat/completions"

    @property
    def total_inflight(self) -> int:
        return sum(p.inflight for p in self.programs.values())

    @property
    def paused_count(self) -> int:
        return sum(1 for p in self.programs.values() if p.paused)


class MultiBackendRouter:
    def __init__(self, backend_urls: List[str]) -> None:
        self.backends: Dict[str, BackendState] = {
            url: BackendState(url=url) for url in backend_urls
        }
        self.program_affinity: Dict[str, str] = {}  # program_id -> backend_url
        
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
        usage_vals: list[float] = []
        running_vals: list[float] = []
        waiting_vals: list[float] = []
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            if line.startswith("vllm:kv_cache_usage_perc"):
                val = self._parse_metric_value(line)
                if val is not None:
                    usage_vals.append(val)
            elif line.startswith("vllm:num_requests_running"):
                val = self._parse_metric_value(line)
                if val is not None:
                    running_vals.append(val)
            elif line.startswith("vllm:num_requests_waiting"):
                val = self._parse_metric_value(line)
                if val is not None:
                    waiting_vals.append(val)
        usage = max(usage_vals) if usage_vals else None
        running = sum(running_vals) if running_vals else None
        waiting = sum(waiting_vals) if waiting_vals else None
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

        if usage >= PAUSE_AT_USAGE:
            backend.high_usage_hits += 1
            if backend.high_usage_hits < 2:
                return
            if now - backend.last_pause_time < 2.5:
                return
            backend.last_pause_time = now
            self._pause_programs_on_backend(backend, usage)

        elif usage < RESUME_AT_USAGE:
            backend.high_usage_hits = 0
            if now - backend.last_resume_time < 10.0:
                return
            backend.last_resume_time = now
            self._resume_programs_on_backend(backend, usage)
        else:
            backend.high_usage_hits = 0

    def _pause_programs_on_backend(self, backend: BackendState, usage: float):
        """Pause low-priority programs on this backend."""
        active = [
            (pid, state)
            for pid, state in backend.programs.items()
            if not state.paused and not state.pause_requested
        ]
        if not active:
            return

        tokens_to_pause = KV_CACHE_TOKEN_BUDGET * 0.01
        active.sort(key=lambda item: (-item[1].step_count, item[1].est_tokens))

        paused_tokens = 0
        newly_paused = []

        for i in range(len(active) - 1, -1, -1):
            if paused_tokens >= tokens_to_pause:
                break

            pid, state = active[i]
            if state.inflight > 0:
                state.pause_requested = True
            else:
                state.paused = True
                state.resume_event.clear()

            paused_tokens += state.est_tokens
            newly_paused.append(pid)

        if newly_paused:
            logger.info(
                f"[{backend.url}] Paused {len(newly_paused)} programs "
                f"(usage={usage:.2%}, tokens={paused_tokens}): {newly_paused}"
            )

    def _resume_programs_on_backend(self, backend: BackendState, usage: float):
        """Resume paused programs on this backend."""
        if usage < 0.5:
            step_pct = 0.20
        elif usage < 0.6:
            step_pct = 0.10
        elif usage < 0.75:
            step_pct = 0.05
        elif usage < 0.8:
            step_pct = 0.03
        elif usage < 0.85:
            step_pct = 0.02
        else:
            step_pct = 0.01

        max_resume_tokens = KV_CACHE_TOKEN_BUDGET * step_pct

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
        backend = self._pick_least_loaded_backend()
        self.program_affinity[program_id] = backend.url
        logger.debug(f"Assigned {program_id} to {backend.url}")
        return backend

    def _pick_least_loaded_backend(self) -> BackendState:
        """Select a healthy backend, prefer those with no paused programs."""
        healthy = [b for b in self.backends.values() if b.healthy]
        if not healthy:
            # Fallback: use any backend
            logger.warning("No healthy backends, using first available")
            return list(self.backends.values())[0]

        candidates = [b for b in healthy if b.paused_count == 0] or healthy

        def score(b: BackendState) -> float:
            return b.running_requests + b.waiting_requests * 4.0

        return min(candidates, key=score)

    # -------------------------------------------------------------------------
    # Token Estimation
    # -------------------------------------------------------------------------

    def estimate_tokens(self, text: str, model_name: str) -> int:
        try:
            if model_name not in _TOKENIZER_CACHE:
                _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
            tok = _TOKENIZER_CACHE[model_name]
            return len(tok.encode(text, add_special_tokens=False))
        except Exception as exc:
            logger.warning(f"Fallback token estimate for {model_name}: {exc}")
            return len(text) // 4

    # -------------------------------------------------------------------------
    # Request Proxying
    # -------------------------------------------------------------------------

    @staticmethod
    def _filtered_headers(headers: httpx.Headers) -> Dict[str, str]:
        hop_by_hop = {"content-length", "transfer-encoding", "connection"}
        return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}

    async def proxy_request(self, backend: BackendState, payload: Dict[str, Any]) -> Response:
        url = backend.completions_url

        if payload.get("stream"):
            resp_cm = self.client.stream("POST", url, json=payload)
            resp = await resp_cm.__aenter__()
            headers = self._filtered_headers(resp.headers)
            status = resp.status_code
            media_type = resp.headers.get("content-type")

            async def iterator():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp_cm.__aexit__(None, None, None)

            return StreamingResponse(
                iterator(),
                status_code=status,
                headers=headers,
                media_type=media_type,
            )

        resp = await self.client.post(url, json=payload)
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


def _get_program_id(payload: Dict[str, Any], request: Request) -> str:
    if "x-program-id" in request.headers:
        return request.headers["x-program-id"]
    if "router_program_id" in payload:
        return str(payload["router_program_id"])
    if "job_id" in payload:
        return str(payload["job_id"])
    return "default"


@app.post("/v1/chat/completions")
async def route_chat_completions(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    program_id = _get_program_id(payload, request)
    model_name = payload.get("model", "default")

    # Extract prompt text for token estimation
    prompt_text = ""
    if "messages" in payload:
        for m in payload["messages"]:
            prompt_text += str(m.get("content", ""))
    elif "prompt" in payload:
        p = payload["prompt"]
        prompt_text = p if isinstance(p, str) else "".join(str(x) for x in p)

    input_tokens = router.estimate_tokens(prompt_text, model_name)

    extra_body = payload.get("extra_body", {})
    is_last_step = extra_body.get("is_last_step", False) if extra_body else False

    # Get the backend for this program (sticky routing)
    backend = router._get_backend_for_program(program_id)

    first_pass = True
    while True:
        async with backend.lock:
            if program_id not in backend.programs:
                backend.programs[program_id] = ProgramState(
                    context_len=input_tokens, step_count=0
                )
            state = backend.programs[program_id]
            state.context_len = input_tokens

            if first_pass:
                step_val = None
                if isinstance(extra_body, dict):
                    raw_step = extra_body.get("step")
                    if isinstance(raw_step, int) and raw_step >= 0:
                        step_val = raw_step
                if step_val is not None:
                    state.step_count = max(state.step_count, step_val)
                else:
                    state.step_count += 1
                first_pass = False

            if not state.paused:
                state.inflight += 1
                break

            wait_event = state.resume_event

        await wait_event.wait()

    try:
        return await router.proxy_request(backend, payload)
    finally:
        async with backend.lock:
            if program_id in backend.programs:
                state = backend.programs[program_id]
                state.inflight = max(0, state.inflight - 1)

                if is_last_step:
                    del backend.programs[program_id]
                    # Clean up affinity
                    if program_id in router.program_affinity:
                        del router.program_affinity[program_id]
                    logger.info(f"[{backend.url}] Program {program_id} completed")
                elif state.pause_requested and state.inflight == 0:
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
