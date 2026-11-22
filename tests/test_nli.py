import os

import pytest

from rag_auditor.auditor import audit_answer

RETURNS = "After dispatch, customers may return unopened products within 7 days for a full refund."
SHIPPING = "Standard shipping takes 3 to 5 business days. Express shipping delivers within 1 business day."
WARRANTY = "Every Acme robot is covered by a limited warranty of 24 months from the date of purchase."


class FakeNLI:
    """Scripted NLI scorer: maps (premise substring, hypothesis substring) to probabilities."""

    def __init__(self, rules):
        self.rules = rules
        self.calls = []

    def score(self, pairs):
        self.calls.append(pairs)
        out = []
        for premise, hyp in pairs:
            probs = {"entailment": 0.05, "neutral": 0.9, "contradiction": 0.05}
            for (p_sub, h_sub), value in self.rules.items():
                if p_sub in premise and h_sub in hyp:
                    probs = value
            out.append(probs)
        return out


ENTAILS = {"entailment": 0.93, "neutral": 0.05, "contradiction": 0.02}
CONTRADICTS = {"entailment": 0.02, "neutral": 0.08, "contradiction": 0.9}


def test_without_nli_behaviour_is_unchanged():
    rep = audit_answer("Unopened items can be sent back for a refund inside a week [1].", [RETURNS])
    assert rep.claims[0].entailment is None and rep.claims[0].contradiction is None


def test_nli_rescues_a_faithful_paraphrase():
    claim = "Unopened items can be sent back for a refund inside a week [1]."
    lexical = audit_answer(claim, [RETURNS]).claims[0]
    assert lexical.verdict != "supported"  # word overlap alone under-scores the paraphrase
    nli = FakeNLI({("unopened products", "sent back"): ENTAILS})
    rep = audit_answer(claim, [RETURNS], nli=nli).claims[0]
    assert rep.verdict == "supported" and rep.entailment == 0.93
    assert rep.best_evidence == RETURNS


def test_nli_catches_a_contradiction_without_negation_words():
    claim = "Standard shipping is faster than express shipping [1]."
    assert audit_answer(claim, [SHIPPING]).claims[0].verdict != "unsupported"  # lexically it looks fine
    nli = FakeNLI({("Express shipping", "faster than express"): CONTRADICTS})
    rep = audit_answer(claim, [SHIPPING], nli=nli).claims[0]
    assert rep.verdict == "unsupported"
    assert any("contradicted by evidence" in i for i in rep.issues)
    assert rep.contradiction == 0.9


def test_exact_lexical_checks_are_not_overridden_by_nli():
    # Wrong number: even if a model were fooled into "entailment", the numeric check still wins.
    nli = FakeNLI({("warranty", "warranty"): ENTAILS})
    rep = audit_answer("Acme robots carry a 36 months warranty [1].", [WARRANTY], nli=nli).claims[0]
    assert rep.verdict == "unsupported"
    assert any("36" in i for i in rep.issues)


def test_nli_only_runs_on_cited_claims_and_few_premises():
    nli = FakeNLI({})
    audit_answer("Shipping takes days. It is fast [1].", [SHIPPING], nli=nli)
    assert len(nli.calls) == 1  # the uncited first claim is not sent to the model
    assert len(nli.calls[0]) <= 3  # at most the 3 most relevant evidence sentences


@pytest.mark.skipif(os.getenv("NLI_MODEL_TESTS") != "1", reason="set NLI_MODEL_TESTS=1 to download and run the real NLI model")
def test_real_cross_encoder():
    from rag_auditor.nli import CrossEncoderNLI

    nli = CrossEncoderNLI()
    para = audit_answer("Unopened items can be sent back for a refund inside a week [1].", [RETURNS], nli=nli).claims[0]
    contra = audit_answer("Standard shipping is faster than express shipping [1].", [SHIPPING], nli=nli).claims[0]
    assert para.entailment > 0.5
    assert contra.verdict == "unsupported"
