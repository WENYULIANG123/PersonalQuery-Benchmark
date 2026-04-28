#!/usr/bin/env python3
"""Stage 4 句法硬校验模块。

LLM 只负责提出候选错误；本模块使用 spaCy 对候选进行严格句法校验，
确保错误词真实落在允许的 ACL / CCOMP 句法区域中。
"""

from __future__ import annotations

import re
import threading
from typing import List, Optional, Tuple

import spacy


_NLP = None
_NLP_LOCK = threading.Lock()

ACL_REGION_TYPES = {"acl", "relcl", "advcl"}
CCOMP_REGION_TYPES = {"ccomp", "modal", "complement_link", "clause_boundary"}
CCOMP_HEAD_DEPS = {"ccomp", "xcomp", "csubj", "csubjpass"}
COMPLEMENT_MARKERS = {"that", "if", "whether"}
MODAL_WORDS = {
    "would", "could", "should", "might", "may", "will", "shall", "must", "ought", "used"
}
ACL_MODIFIER_DEPS = {"amod", "acomp", "advmod", "oprd"}
ACL_NP_DEPS = {"compound", "nmod", "poss", "nsubj", "dobj", "obj", "pobj", "attr"}


def _load_nlp():
    global _NLP
    if _NLP is None:
        with _NLP_LOCK:
            if _NLP is None:
                _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def _find_case_insensitive(text: str, pattern: str) -> Optional[int]:
    match = re.search(re.escape(pattern), text, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.start()


def _replace_single_word(sentence_text: str, original: str, corrected: str) -> Tuple[str, int]:
    pattern = re.compile(rf"\b{re.escape(original)}\b", flags=re.IGNORECASE)
    match = pattern.search(sentence_text)
    if match is None:
        raise ValueError(f"Original word '{original}' not found in sentence: {sentence_text}")
    replaced = sentence_text[:match.start()] + corrected + sentence_text[match.end():]
    return replaced, match.start()


class SyntaxHardValidator:
    """对 Stage 4 的候选错误进行 spaCy 句法硬校验。"""

    def __init__(self):
        self.nlp = _load_nlp()

    def validate_candidate(
        self,
        review_text: str,
        span_text: str,
        original: str,
        corrected: str,
        error_category: str,
        region_type: str,
        error_type: str,
    ) -> Tuple[bool, str]:
        if error_category == "acl" and region_type not in ACL_REGION_TYPES:
            return False, "invalid_acl_region_type"
        if error_category == "ccomp" and region_type not in CCOMP_REGION_TYPES:
            return False, "invalid_ccomp_region_type"

        sentence_text = self._locate_sentence(review_text, span_text, original)
        if sentence_text is None:
            return False, "syntax_sentence_not_found"

        corrected_sentence, replace_char = _replace_single_word(sentence_text, original, corrected)
        doc = self.nlp(corrected_sentence)
        anchor = self._locate_anchor_token(doc, corrected.lower(), replace_char)
        if anchor is None:
            return False, "syntax_anchor_not_found"

        if error_category == "acl":
            return self._validate_acl(anchor, region_type, error_type)
        return self._validate_ccomp(doc, anchor, region_type, error_type)

    def _locate_sentence(self, review_text: str, span_text: str, original: str) -> Optional[str]:
        doc = self.nlp(review_text)

        normalized_span = _normalize_space(span_text)
        if normalized_span:
            offset = _find_case_insensitive(review_text, normalized_span)
            if offset is not None:
                end = offset + len(normalized_span)
                for sent in doc.sents:
                    if sent.start_char <= offset < sent.end_char or sent.start_char < end <= sent.end_char:
                        return sent.text

        pattern = re.compile(rf"\b{re.escape(original)}\b", flags=re.IGNORECASE)
        match = pattern.search(review_text)
        if match is None:
            return None

        start = match.start()
        for sent in doc.sents:
            if sent.start_char <= start < sent.end_char:
                return sent.text
        return None

    def _locate_anchor_token(self, doc, corrected_lower: str, replace_char: int):
        candidates = [token for token in doc if token.text.lower() == corrected_lower]
        if not candidates:
            return None
        return min(candidates, key=lambda token: abs(token.idx - replace_char))

    def _validate_acl(self, anchor, region_type: str, error_type: str) -> Tuple[bool, str]:
        matching_heads = [token for token in anchor.doc if token.dep_ == region_type]
        if not matching_heads:
            return False, "syntax_no_matching_acl_head"

        for head in matching_heads:
            if not self._is_valid_acl_head(head, region_type):
                continue

            if anchor != head and anchor not in list(head.subtree):
                continue

            if anchor == head:
                return True, "ok"

            if error_type in {"attribute_typo", "modifier_typo"}:
                if anchor.pos_ in {"ADJ", "ADV"} and anchor.dep_ in ACL_MODIFIER_DEPS:
                    return True, "ok"
                continue

            if error_type == "np_inflection":
                if anchor.pos_ in {"NOUN", "PROPN", "PRON"} and anchor.dep_ in ACL_NP_DEPS:
                    return True, "ok"
                continue

            raise ValueError(f"Unsupported acl error_type: {error_type}")

        if error_type in {"attribute_typo", "modifier_typo"}:
            return False, "syntax_acl_modifier_role_mismatch"
        if error_type == "np_inflection":
            return False, "syntax_acl_np_role_mismatch"
        return False, "syntax_acl_subtree_mismatch"

    def _is_valid_acl_head(self, head, region_type: str) -> bool:
        if region_type in {"acl", "relcl"}:
            return True
        if region_type == "advcl":
            return any(child.dep_ == "mark" for child in head.children)
        raise ValueError(f"Unsupported acl region_type: {region_type}")

    def _validate_ccomp(self, doc, anchor, region_type: str, error_type: str) -> Tuple[bool, str]:
        ccomp_heads = [token for token in doc if token.dep_ in CCOMP_HEAD_DEPS]

        if region_type == "ccomp":
            if not ccomp_heads:
                return False, "syntax_no_ccomp_head"
            for head in ccomp_heads:
                if anchor == head or anchor in list(head.subtree):
                    return True, "ok"
            if error_type == "clause_shell_typo":
                for head in ccomp_heads:
                    if head.head == anchor or anchor in list(head.ancestors):
                        return True, "ok"
            return False, "syntax_ccomp_subtree_mismatch"

        if region_type == "modal":
            if anchor.text.lower() not in MODAL_WORDS:
                return False, "syntax_modal_word_mismatch"
            if anchor.tag_ != "MD":
                return False, "syntax_modal_tag_mismatch"
            for head in ccomp_heads:
                if anchor == head or anchor in list(head.subtree):
                    return True, "ok"
            return False, "syntax_modal_outside_ccomp"

        if region_type == "complement_link":
            if anchor.text.lower() not in COMPLEMENT_MARKERS:
                return False, "syntax_complement_marker_mismatch"
            if anchor.dep_ != "mark":
                return False, "syntax_complement_marker_dep_mismatch"
            if anchor.head.dep_ in CCOMP_HEAD_DEPS:
                return True, "ok"
            return False, "syntax_complement_marker_outside_ccomp"

        if region_type == "clause_boundary":
            if not ccomp_heads:
                return False, "syntax_no_ccomp_head"
            if anchor.pos_ not in {"VERB", "AUX", "SCONJ"} and error_type != "clause_boundary_error":
                return False, "syntax_clause_boundary_pos_mismatch"
            for head in ccomp_heads:
                if anchor == head or anchor in list(head.subtree):
                    return True, "ok"
            return False, "syntax_clause_boundary_outside_ccomp"

        raise ValueError(f"Unsupported ccomp region_type: {region_type}")
