"""Seeded, fictional benchmark fixtures with independently stored answers."""
from __future__ import annotations

import random
from collections.abc import Callable

from .models import Fixture

SUITES = ("needle", "retention", "distractor", "multi_fact", "unanswerable")
NAMES = ("Mira Chen", "Amina Yusuf", "Nadia Ali", "Omar Shah", "Farah Noor")
WORDS = ("amber", "cedar", "harbor", "meadow", "silver", "violet", "willow")


def generate_fixtures(seed: int = 42, per_suite: int = 2) -> list[Fixture]:
    """Generate deterministic tests. Prompt token sizing is performed by the runtime."""
    rng = random.Random(seed)
    out: list[Fixture] = []
    for i in range(per_suite):
        incident = f"AX-{rng.randrange(1000, 9999)}"
        code = f"{rng.choice(WORDS).upper()}-{rng.randrange(100, 999)}"
        fact = f"Incident {incident} was assigned to engineer {rng.choice(NAMES)}. Recovery code: {code}."
        filler = [f"Daily operations note {j}: {rng.choice(WORDS)} team reviewed routine service telemetry." for j in range(18)]
        position = ("beginning", "25%", "middle", "75%", "end")[i % 5]
        insert_at = {"beginning": 0, "25%": 4, "middle": 9, "75%": 14, "end": 18}[position]
        doc = "\n".join(filler[:insert_at] + [fact] + filler[insert_at:])
        out.append(Fixture(id=f"needle_{i+1:03}", suite="needle", seed=seed, document=doc,
                           question=f"What is the recovery code for incident {incident}?",
                           expected_answer=code, fact_position=position, fact_spans=[fact]))

        project = f"Project {rng.choice(('Aster', 'Orion', 'Juniper', 'Solstice'))}-{i}"
        region = f"{rng.choice(('eu', 'us', 'ap'))}-{rng.choice(('central', 'east', 'south'))}-{rng.randrange(1, 4)}"
        release = f"REL-{rng.randrange(100, 999)}"
        facts = [f"{project} deploys in {region}.", f"Release coordinator is {rng.choice(NAMES)}.", f"Rollback code is {release}."]
        retention_questions = (("region", f"Which deployment region is recorded for {project}?", region, facts[0]),
                               ("coordinator", f"Who coordinates the release for {project}?", facts[1].split(" is ")[-1].rstrip("."), facts[1]),
                               ("rollback", f"What rollback code is recorded for {project}?", release, facts[2]))
        for suffix, question, expected, span in retention_questions:
            out.append(Fixture(id=f"retention_{i+1:03}_{suffix}", suite="retention", seed=seed,
                               document=" ".join(facts), question=question, expected_answer=expected,
                               fact_spans=[span], facts=facts))

        records = [(f"C-{n}", plan) for n, plan in ((104, "SILVER"), (140, "GOLD"), (401, "BASIC"))]
        target = records[i % len(records)]
        distractors = "\n".join(f"Customer {cid} uses plan {plan}." for cid, plan in records)
        out.append(Fixture(id=f"distractor_{i+1:03}", suite="distractor", seed=seed, document=distractors,
                           question=f"Which plan belongs to customer {target[0]}?", expected_answer=target[1],
                           distractors=2, fact_spans=[f"Customer {target[0]} uses plan {target[1]}."]))

        owner, service = rng.choice(NAMES).split()[0], f"svc-{rng.choice(WORDS)}-{i}"
        out.append(Fixture(id=f"multi_{i+1:03}", suite="multi_fact", seed=seed,
                           document=f"Engineer {owner} owns service {service}. Service {service} runs in {region}.",
                           question="Return the owner, service, and hosting region, separated by |.",
                           expected_answer=f"{owner}|{service}|{region}",
                           fact_spans=[f"Engineer {owner} owns service {service}.", f"Service {service} runs in {region}."],
                           facts=[owner, service, region], required_facts=[owner, service, region], answer_type="multi"))

        out.append(Fixture(id=f"unanswerable_{i+1:03}", suite="unanswerable", seed=seed,
                           document=f"Project {project} launched on June 8. Deployment approved by {rng.choice(NAMES)}.",
                           question="What is the project's rollback authorization code?", expected_answer="NOT_FOUND",
                           answer_type="unanswerable"))
    return out


def resize_fixtures(fixtures: list[Fixture], tokenizer: Callable[[str], list[int]], *,
                    context: int, output_budget: int, target_fraction: float = 0.72,
                    tolerance: float = 0.05,
                    prompt_builder: Callable[[str, str], str] | None = None,
                    token_position: Callable[[str, str], int] | None = None) -> list[Fixture]:
    """Resize each prompt with model-tokenized fictional filler and place facts by token offset.

    The target applies to the complete serialized prompt, including instructions and question.
    If it cannot fit within tolerance without exceeding the context/output budget, the fixture
    carries an invalid_reason and is excluded by the runner.
    """
    from .runtime import build_prompt
    build = prompt_builder or build_prompt
    result: list[Fixture] = []
    for original in fixtures:
        target = max(1, int(context * target_fraction))
        usable = context - output_budget
        spans = list(original.fact_spans)
        core = "\n".join(spans)
        remainder = original.document
        for span in spans:
            remainder = remainder.replace(span, "")
        remainder = remainder.strip()
        if not core:
            core, remainder = original.document, ""
        # Each filler is a fictional operational note. No answer values are interpolated.
        def filler(i: int) -> str:
            return (f"Operations journal entry {i}: service health remained within the reviewed "
                    "deployment window; the on-call team recorded routine telemetry and closed "
                    "the maintenance note after verification.")

        position_pct = {"beginning": 0.0, "25%": 0.25, "middle": 0.5,
                        "75%": 0.75, "end": 1.0}.get(original.fact_position or "middle", 0.5)
        if original.answer_type == "unanswerable":
            position_pct = 1.0

        def materialize(
            count: int,
            *,
            fact_core: str = core,
            other_context: str = remainder,
            insertion_fraction: float = position_pct,
        ) -> str:
            all_fill = [filler(i) for i in range(count)]
            split = min(count, round(count * insertion_fraction))
            blocks = (
                all_fill[:split]
                + ([fact_core] if fact_core else [])
                + ([other_context] if other_context else [])
                + all_fill[split:]
            )
            doc = "\n".join(blocks)
            if not doc:
                doc = filler(0)
            return doc

        low, high, best = 0, 1, ("", 0, 0)
        while high < 4096:
            doc = materialize(high)
            n = len(tokenizer(build(doc, original.question)))
            if n >= target or n >= usable:
                break
            low, high = high, high * 2
        while low <= high:
            mid = (low + high) // 2
            doc = materialize(mid)
            count = len(tokenizer(build(doc, original.question)))
            if count <= usable:
                best = (doc, count, 0)
                if count < target:
                    low = mid + 1
                else:
                    high = mid - 1
            else:
                high = mid - 1
        doc, token_count, fact_offset = best
        marker = spans[0] if spans else ""
        if marker:
            serialized = build(doc, original.question)
            if token_position:
                fact_offset = token_position(serialized, marker)
            else:
                marker_offset = serialized.find(marker)
                fact_offset = len(tokenizer(serialized[:marker_offset])) if marker_offset >= 0 else 0
        invalid = None
        if token_count == 0 or token_count + output_budget > context:
            invalid = f"Serialized prompt plus {output_budget} output tokens exceeds context {context}"
        elif abs(token_count - target) > max(1, int(target * tolerance)):
            invalid = f"Could not reach {target} target prompt tokens within {tolerance:.0%} tolerance; got {token_count}"
        position = (f"{100 * fact_offset / token_count:.1f}%" if spans and token_count else None)
        result.append(original.model_copy(update={"document": doc, "target_context_length": context,
                                                   "actual_token_length": token_count,
                                                   "fact_position": position,
                                                   "invalid_reason": invalid}))
    return result
