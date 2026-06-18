import json, os as os_mod, time

MAX_RETRIES = 3

# ── Module-level imports ────────────────────────────────────
try:
    from reference import analyze as _ref_analyze, report as _ref_report
    _HAS_REFERENCE = True
except ImportError:
    _HAS_REFERENCE = False

# ── Success detection ────────────────────────────────────────
FAILURE_MARKERS = [
    "max iterations",
    "task may be incomplete",
    "error:",
    "cancelled by user",
    "unknown tool",
    "i couldn't",
    "i was unable",
    "failed to",
    "could not complete",
]

_REF_MAP = {
    "database|sql|prisma|postgres": "https://github.com/prisma/prisma-examples",
    "docker|container":             "https://github.com/docker/awesome-compose",
    "react|vue|nextjs|frontend":   "https://github.com/vercel/next.js",
    "api|rest|graphql":             "https://github.com/public-apis/public-apis",
    "python|flask|django":          "https://github.com/pallets/flask",
    "go|golang":                    "https://github.com/golang/go",
    "rust|cargo":                   "https://github.com/rust-lang/rust",
    "git|github|ci":                "https://github.com/actions/starter-workflows",
}


def _is_step_success(final: str | None) -> bool:
    if not final or len(final) < 5:
        return False
    lo = final.lower()
    return not any(marker in lo for marker in FAILURE_MARKERS)


def _get_ref_url(step_desc: str) -> str | None:
    lo = step_desc.lower()
    for keywords, url in _REF_MAP.items():
        if any(kw in lo for kw in keywords.split("|")):
            return url
    return None


class AttemptRecord:
    def __init__(self, step_idx, tool, args, result, success):
        self.step_idx = step_idx
        self.tool = tool
        self.args = args
        self.result = result[:200] if result else ""
        self.success = success
        self.timestamp = time.time()

class AttemptHistory:
    def __init__(self):
        self.records = []

    def add(self, step_idx, tool, args, result, success):
        self.records.append(AttemptRecord(step_idx, tool, args, result, success))

    def get_for_step(self, step_idx):
        return [r for r in self.records if r.step_idx == step_idx]

    def failures_for_step(self, step_idx):
        return [r for r in self.records if r.step_idx == step_idx and not r.success]

    def summary(self):
        if not self.records:
            return ""
        lines = ["## Attempt History"]
        for r in self.records:
            status = "OK" if r.success else "FAIL"
            lines.append(f"- Step {r.step_idx}: {r.tool} -> {status}")
        return "\n".join(lines)

    def failure_context(self, step_idx):
        fails = self.failures_for_step(step_idx)
        if not fails:
            return ""
        lines = ["## Previous Attempts (Failed)"]
        for i, r in enumerate(fails, 1):
            lines.append(f"Attempt {i}: {r.tool}({json.dumps(r.args)[:80]})")
            lines.append(f"  Result: {r.result[:150]}")
            lines.append(f"  -> FAILED")
        lines.append("Try a DIFFERENT approach this time.")
        return "\n".join(lines)

class StepStatus:
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"

class Plan:
    def __init__(self, goal, steps=None):
        self.goal = goal
        self.steps = steps or []
        self.current_idx = 0

    def add_step(self, description):
        self.steps.append({"desc": description, "status": StepStatus.PENDING})

    def current(self):
        if self.current_idx < len(self.steps):
            return self.steps[self.current_idx]
        return None

    def mark_done(self):
        if self.current():
            self.current()["status"] = StepStatus.DONE

    def mark_failed(self):
        if self.current():
            self.current()["status"] = StepStatus.FAILED

    def advance(self):
        self.current_idx += 1

    def is_complete(self):
        return self.current_idx >= len(self.steps)

    def progress_str(self):
        done = sum(1 for s in self.steps if s["status"] == StepStatus.DONE)
        failed = sum(1 for s in self.steps if s["status"] == StepStatus.FAILED)
        total = len(self.steps)
        icons = {
            StepStatus.PENDING:     "   ",
            StepStatus.IN_PROGRESS: ">> ",
            StepStatus.DONE:        " \u2713 ",  # checkmark
            StepStatus.FAILED:      " \u2717 ",  # x mark
        }
        parts = []
        for i, s in enumerate(self.steps):
            icon = icons.get(s["status"], "   ")
            desc = s.get("desc", "")[:50]
            parts.append(f"  {icon} Step {i+1}: {desc}")
        return "\n".join(parts)

class PlannerAgent:
    def __init__(self, agent_session):
        self.session = agent_session
        self.history = AttemptHistory()
        self.plan = None

    def run_with_plan(self, user_input):
        """Plan -> Execute -> Retry -> Complete"""
        # Phase 1: Plan
        plan = self._create_plan(user_input)
        yield {"type": "plan", "goal": user_input, "steps": plan.steps}

        # Phase 2: Execute per step
        for step_idx in range(len(plan.steps)):
            step = plan.steps[step_idx]
            retries = 0
            success = False

            while retries <= MAX_RETRIES and not success:
                yield {"type": "step_start", "index": step_idx, "desc": step["desc"], "attempt": retries + 1}

                # Build step prompt with context
                step_prompt = self._build_step_prompt(step, step_idx)

                # Execute via agent
                final = None
                try:
                    for event in self.session.run_stream(step_prompt):
                        yield event
                        if event["type"] == "final":
                            final = event["content"]
                except Exception as e:
                    final = f"Error: {e}"

                # Check if result indicates success
                if _is_step_success(final):
                    success = True
                    plan.mark_done()
                    plan.advance()
                    self.history.add(step_idx, "agent", {}, final, True)
                    yield {"type": "step_done", "index": step_idx}
                else:
                    retries += 1
                    self.history.add(step_idx, "agent", {}, final, False)
                    if retries <= MAX_RETRIES:
                        yield {"type": "step_retry", "index": step_idx, "retry": retries}

            if not success and not retries <= MAX_RETRIES:
                plan.mark_failed()
                plan.advance()
                yield {"type": "step_failed", "index": step_idx}
                # Auto-analyze reference for learning
                if _HAS_REFERENCE:
                    ref_url = _get_ref_url(step.get("desc", ""))
                    if ref_url:
                        try:
                            data = _ref_analyze(ref_url)
                            if "error" not in data:
                                yield {"type": "token", "content": "\n[auto] Learning from: " + ref_url + "\n"}
                        except:
                            pass

        # Phase 3: Summary
        summary = self._build_summary(plan)
        yield {"type": "final", "content": summary}

    def _create_plan(self, goal):
        plan = Plan(goal)
        prompt = f"Break down this task into 1-4 clear steps:\n{goal}\n\nNumber each step with Step N: ..."
        result = self.session.provider.chat([
            {"role": "system", "content": "You are a planner. Break tasks into numbered steps."},
            {"role": "user", "content": prompt}
        ])
        import re
        for match in re.finditer(r"Step \d+[.:]\s*(.+?)(?=Step \d+[.:]|$)", result, re.DOTALL):
            desc = match.group(1).strip()
            if desc:
                plan.add_step(desc)
        if not plan.steps:
            plan.add_step(result.strip()[:100])
        return plan

    def _build_step_prompt(self, step, step_idx):
        ctx = self.history.failure_context(step_idx)
        step_desc = step.get("desc", "")
        prompt = f"[STEP {step_idx + 1}] {step_desc}"
        if ctx:
            prompt += f"\n\n{ctx}"
        return prompt

    def _build_summary(self, plan):
        done = sum(1 for s in plan.steps if s["status"] == StepStatus.DONE)
        total = len(plan.steps)
        return f"Plan complete: {done}/{total} steps done.\n\n{plan.progress_str()}"