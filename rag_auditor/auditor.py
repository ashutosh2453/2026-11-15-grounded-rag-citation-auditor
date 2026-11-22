"""Claim-level citation auditor.

Splits an answer into claims (sentences), resolves each claim's ``[n]``
citations, and measures how well the cited evidence supports it:

* token recall      - fraction of the claim's content words present in evidence
* ROUGE-L recall    - longest-common-subsequence overlap with the best evidence sentence
* numeric check     - every number in the claim must appear in the cited evidence
* negation check    - polarity of the claim must match the best evidence sentence

Claims are then labelled ``supported``, ``partially_supported``,
``unsupported``, ``misattributed`` (a different source supports it),
``uncited`` or ``non_claim``. The report aggregates these into a
groundedness score and a citation accuracy score.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from .text import content_tokens, has_negation, lcs_length, numbers, split_sentences

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
ABSTAIN_RE = re.compile(
    r"(could not find|couldn't find|not (?:mentioned|provided|available|stated|contain)|"
    r"no information|do(?:es)? not know|don't know|insufficient (?:information|context))",
    re.IGNORECASE,
)

SUPPORTED, PARTIAL = 0.70, 0.40
NUMERIC_CAP = 0.30


@dataclass
class EvidenceMatch:
    source: int
    score: float
    sentence: str
    issues: list[str] = field(default_factory=list)


@dataclass
class ClaimAudit:
    claim: str
    citations: list[int]
    verdict: str
    support_score: float
    best_evidence: str | None
    best_source: int | None
    suggested_source: int | None
    issues: list[str]
    entailment: float | None = None  # NLI probabilities, only when an NLI verifier is used
    contradiction: float | None = None


@dataclass
class AuditReport:
    overall: str
    groundedness: float
    citation_accuracy: float | None
    verdict_counts: dict[str, int]
    claims: list[ClaimAudit]

    def to_dict(self) -> dict:
        return asdict(self)


def split_claims(answer: str) -> list[tuple[str, list[int]]]:
    """Return ``(claim_text_without_citations, [citation numbers])`` pairs.

    Handles both ``Claim [1].`` and ``Claim. [1]`` styles; a fragment made only
    of citations is attached to the previous claim.
    """
    claims: list[tuple[str, list[int]]] = []
    for sentence in split_sentences(answer):
        cites = [int(n) for group in CITATION_RE.findall(sentence) for n in re.split(r"\s*,\s*", group)]
        text = CITATION_RE.sub("", sentence)
        text = re.sub(r"\s+([.!?,;:])", r"\1", text).strip()
        if not re.search(r"\w", text):
            if claims and cites:
                prev_text, prev_cites = claims[-1]
                claims[-1] = (prev_text, prev_cites + cites)
            continue
        claims.append((text, list(dict.fromkeys(cites))))
    return claims


def _score_against(claim: str, evidence_text: str, source: int) -> EvidenceMatch:
    claim_toks = content_tokens(claim)
    claim_set = set(claim_toks)
    best_score, best_sentence = 0.0, ""
    for ev in split_sentences(evidence_text) or [evidence_text]:
        ev_toks = content_tokens(ev)
        recall = len(claim_set & set(ev_toks)) / len(claim_set)
        rouge = lcs_length(claim_toks, ev_toks) / len(claim_toks)
        score = 0.6 * recall + 0.4 * rouge
        if score > best_score:
            best_score, best_sentence = score, ev
    # A claim may legitimately merge facts from several sentences of one chunk.
    chunk_recall = len(claim_set & set(content_tokens(evidence_text))) / len(claim_set)
    score = max(best_score, 0.85 * chunk_recall)

    issues: list[str] = []
    missing_numbers = numbers(claim) - numbers(evidence_text)
    if missing_numbers:
        issues.append("numbers not found in evidence: " + ", ".join(sorted(missing_numbers)))
        score = min(score, NUMERIC_CAP)
    if best_sentence and has_negation(claim) != has_negation(best_sentence):
        issues.append("negation mismatch with evidence")
        score *= 0.5
    return EvidenceMatch(source, round(score, 3), best_sentence or evidence_text, issues)


def _label(score: float) -> str:
    if score >= SUPPORTED:
        return "supported"
    if score >= PARTIAL:
        return "partially_supported"
    return "unsupported"


ENTAIL_THRESHOLD = 0.5
CONTRADICT_THRESHOLD = 0.7


def _candidate_sentences(claim: str, evidence: list[str], n: int = 3) -> list[tuple[str, int]]:
    """The n evidence sentences most lexically similar to the claim, with their overlap."""
    claim_set = set(content_tokens(claim))
    scored = [(s, len(claim_set & set(content_tokens(s)))) for text in evidence for s in (split_sentences(text) or [text])]
    scored.sort(key=lambda x: -x[1])
    return scored[:n]


def _apply_nli(nli, claim: str, evidence: list[str], match: EvidenceMatch) -> tuple[EvidenceMatch, float, float]:
    """Combine lexical support with NLI: rescue paraphrases, catch contradictions.

    Lexical numeric/negation checks stay authoritative (they are exact); NLI may raise a
    score only when no such check fired, and may always lower it on a confident contradiction.
    """
    candidates = _candidate_sentences(claim, evidence)
    if not candidates:
        return match, 0.0, 0.0
    premises = [s for s, _ in candidates]
    probs = nli.score([(p, claim) for p in premises])
    best_e = max(range(len(probs)), key=lambda i: probs[i]["entailment"])
    ent = probs[best_e]["entailment"]
    # Contradiction only counts from the sentence(s) most about the claim: an unrelated
    # sentence ("standard takes 3-5 days" vs "express arrives in a day") often looks
    # contradictory to an NLI model without the claim actually being false.
    top_overlap = candidates[0][1]
    con = max(p["contradiction"] for p, (_, ov) in zip(probs, candidates) if ov == top_overlap)
    score, issues, sentence = match.score, list(match.issues), match.sentence
    if con >= CONTRADICT_THRESHOLD and con > ent:
        issues.append(f"contradicted by evidence (NLI p={con:.2f})")
        score = min(score, NUMERIC_CAP)
    elif ent >= ENTAIL_THRESHOLD and not match.issues and ent > score:
        score, sentence = ent, premises[best_e]  # faithful paraphrase that word overlap under-scores
    return EvidenceMatch(match.source, round(score, 3), sentence, issues), round(ent, 3), round(con, 3)


def audit_answer(answer: str, sources: list[str], nli=None) -> AuditReport:
    """Audit ``answer`` against ``sources`` (1-indexed, matching ``[n]`` citations).

    ``nli`` is an optional :class:`rag_auditor.nli.NLIScorer` that adds entailment and
    contradiction checks on top of the lexical rules.
    """
    claims_out: list[ClaimAudit] = []
    for text, cites in split_claims(answer):
        if not content_tokens(text):
            claims_out.append(ClaimAudit(text, cites, "non_claim", 0.0, None, None, None, []))
            continue

        all_matches = [_score_against(text, src, i) for i, src in enumerate(sources, start=1)]
        best_any = max(all_matches, key=lambda m: m.score, default=None)
        issues: list[str] = []
        valid = [c for c in cites if 1 <= c <= len(sources)]
        issues += [f"citation [{c}] does not exist" for c in cites if c not in valid]

        if not valid:
            verdict = "uncited"
            if ABSTAIN_RE.search(text):
                verdict = "non_claim"
            suggested = best_any.source if best_any and best_any.score >= SUPPORTED else None
            claims_out.append(ClaimAudit(text, cites, verdict, best_any.score if best_any else 0.0,
                                         best_any.sentence if best_any else None,
                                         best_any.source if best_any else None, suggested, issues))
            continue

        cited = [all_matches[c - 1] for c in valid]
        best = max(cited, key=lambda m: m.score)
        # Combined evidence lets a claim draw on several cited sources at once.
        combined = _score_against(text, " ".join(sources[c - 1] for c in valid), best.source)
        if combined.score > best.score:
            best = EvidenceMatch(best.source, combined.score, best.sentence, combined.issues)
        ent = con = None
        if nli is not None:
            best, ent, con = _apply_nli(nli, text, [sources[c - 1] for c in valid], best)
        verdict = _label(best.score)
        suggested = None
        if verdict == "unsupported" and best_any and best_any.score >= SUPPORTED and best_any.source not in valid:
            verdict, suggested = "misattributed", best_any.source
        issues += best.issues
        claims_out.append(ClaimAudit(text, cites, verdict, best.score, best.sentence, best.source, suggested, issues, ent, con))

    return _summarise(answer, claims_out)


def _summarise(answer: str, claims: list[ClaimAudit]) -> AuditReport:
    counts = Counter(c.verdict for c in claims)
    factual = [c for c in claims if c.verdict != "non_claim"]
    if not factual:
        overall = "abstained" if ABSTAIN_RE.search(answer) else "no_claims"
        return AuditReport(overall, 1.0 if overall == "abstained" else 0.0, None, dict(counts), claims)

    weights = {"supported": 1.0, "partially_supported": 0.5}
    groundedness = sum(weights.get(c.verdict, 0.0) for c in factual) / len(factual)
    cited = [c for c in factual if c.citations]
    accuracy = (sum(c.verdict in weights for c in cited) / len(cited)) if cited else None
    if groundedness >= 0.9:
        overall = "grounded"
    elif groundedness >= 0.5:
        overall = "partially_grounded"
    else:
        overall = "ungrounded"
    return AuditReport(overall, round(groundedness, 3), None if accuracy is None else round(accuracy, 3),
                       dict(counts), claims)
