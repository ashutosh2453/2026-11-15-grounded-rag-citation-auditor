import pytest

from rag_auditor.auditor import audit_answer, split_claims
from rag_auditor.chunking import chunk_text
from rag_auditor.generation import ABSTAIN_MESSAGE, ExtractiveGenerator
from rag_auditor.retrieval import HybridRetriever
from rag_auditor.store import Chunk, DocumentStore
from rag_auditor.text import content_tokens, has_negation, lcs_length, numbers, split_sentences, stem

WARRANTY = ("Every Acme vacuum robot is covered by a limited warranty of 24 months from the date of purchase. "
            "The warranty covers manufacturing defects in the motor, battery and charging dock.")
EXCLUSIONS = "The warranty does not cover damage caused by water, misuse or unauthorised repairs."
SHIPPING = "Standard shipping within India takes 3 to 5 business days and is free for orders above 2,000 rupees."


# ---------- text ----------

def test_numbers_are_normalised():
    assert numbers("costs 2,000 rupees, 3.50 kg and 70%") == {"2000", "3.5", "70%"}


def test_stem_and_content_tokens():
    assert stem("repairs") == "repair"
    assert stem("batteries") == "battery"
    assert "the" not in content_tokens("The warranty covers the motor")


def test_negation_detection_handles_contractions():
    assert has_negation("It doesn't cover water damage")
    assert not has_negation("It covers water damage")


def test_split_sentences_and_lcs():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert lcs_length(list("abcde"), list("ace")) == 3


# ---------- chunking ----------

def test_chunking_respects_limit_and_paragraphs():
    text = " ".join(f"Sentence number {i} has five words." for i in range(30)) + "\n\nNew paragraph here."
    chunks = chunk_text(text, max_words=30, overlap_sentences=1)
    assert len(chunks) > 3
    assert chunks[-1] == "New paragraph here."
    assert all(len(c.split()) <= 36 for c in chunks)


def test_markdown_headings_do_not_fuse_with_body():
    chunks = chunk_text("# Warranty Policy\nThe warranty lasts 24 months.\n\n## Returns ##\nReturns take 7 days.")
    assert chunks == ["Warranty Policy", "The warranty lasts 24 months.", "Returns", "Returns take 7 days."]


def test_extractive_generator_skips_headings():
    r = HybridRetriever()
    r.build(_chunks("Warranty Policy", "The warranty policy lasts 24 months."))
    gen = ExtractiveGenerator().generate("warranty policy", r.search("warranty policy"))
    assert "Warranty Policy [" not in gen.text
    assert "The warranty policy lasts 24 months [2]." in gen.text


def test_chunking_rejects_bad_args():
    with pytest.raises(ValueError):
        chunk_text("x", max_words=5)


# ---------- retrieval ----------

def _chunks(*texts):
    return [Chunk(i, i, 0, t, f"doc{i}") for i, t in enumerate(texts, start=1)]


def test_hybrid_retrieval_ranks_relevant_chunk_first():
    r = HybridRetriever()
    r.build(_chunks(WARRANTY, SHIPPING, EXCLUSIONS))
    hits = r.search("how many months is the warranty", top_k=3)
    assert hits[0].chunk.text == WARRANTY
    assert all(h.chunk.text != SHIPPING for h in hits)  # no lexical overlap -> not returned


def test_retrieval_empty_query_or_index():
    r = HybridRetriever()
    assert r.search("anything") == []
    r.build(_chunks(WARRANTY))
    assert r.search("the and of") == []
    with pytest.raises(ValueError):
        r.search("warranty", top_k=0)


# ---------- generation ----------

def test_extractive_generator_cites_sources():
    r = HybridRetriever()
    r.build(_chunks(WARRANTY, SHIPPING))
    ctx = r.search("warranty months purchase")
    gen = ExtractiveGenerator().generate("warranty months purchase", ctx)
    assert "[1]" in gen.text and "24 months" in gen.text


def test_extractive_generator_abstains_without_context():
    assert ExtractiveGenerator().generate("warranty?", []).text == ABSTAIN_MESSAGE


# ---------- auditor ----------

def test_split_claims_both_citation_styles():
    claims = split_claims("The warranty lasts 24 months [1]. Water damage is excluded. [2]")
    assert claims == [("The warranty lasts 24 months.", [1]), ("Water damage is excluded.", [2])]


def test_supported_claim():
    rep = audit_answer("Acme robots have a limited warranty of 24 months from purchase [1].", [WARRANTY])
    assert rep.claims[0].verdict == "supported"
    assert rep.overall == "grounded" and rep.citation_accuracy == 1.0


def test_numeric_hallucination_is_flagged():
    rep = audit_answer("Acme robots have a limited warranty of 36 months from purchase [1].", [WARRANTY])
    c = rep.claims[0]
    assert c.verdict == "unsupported"
    assert any("36" in i for i in c.issues)


def test_negation_flip_is_flagged():
    rep = audit_answer("The warranty does cover damage caused by water [1].", [EXCLUSIONS])
    assert rep.claims[0].verdict != "supported"
    assert any("negation" in i for i in rep.claims[0].issues)


def test_misattributed_citation_suggests_correct_source():
    rep = audit_answer("Standard shipping within India takes 3 to 5 business days [1].", [WARRANTY, SHIPPING])
    c = rep.claims[0]
    assert c.verdict == "misattributed" and c.suggested_source == 2


def test_uncited_and_invalid_citations():
    rep = audit_answer("Standard shipping takes 3 to 5 business days. Water damage is excluded [9].", [SHIPPING])
    assert rep.claims[0].verdict == "uncited" and rep.claims[0].suggested_source == 1
    assert rep.claims[1].verdict == "uncited"
    assert "citation [9] does not exist" in rep.claims[1].issues
    assert rep.overall == "ungrounded"


def test_abstention_is_recognised():
    rep = audit_answer(ABSTAIN_MESSAGE, [WARRANTY])
    assert rep.overall == "abstained"


def test_multi_source_claim_uses_combined_evidence():
    a = "The robot warranty is 24 months."
    b = "Shipping is free for orders above 2,000 rupees."
    rep = audit_answer("The robot warranty is 24 months and shipping is free above 2,000 rupees [1][2].", [a, b])
    assert rep.claims[0].verdict == "supported"


# ---------- store ----------

def test_store_roundtrip_and_cascade_delete(tmp_path):
    s = DocumentStore(tmp_path / "t.db")
    doc = s.add_document("w", ["a b c", "d e f"], None, s.fingerprint("x"))
    assert s.find_by_hash(s.fingerprint("x")).chunk_count == 2
    assert len(s.all_chunks()) == 2
    assert s.delete_document(doc.id)
    assert s.all_chunks() == [] and not s.delete_document(doc.id)
    s.close()
