#!/usr/bin/env python3
"""
Extract linguistic features from user reviews, using only sentences with 25-30 words.

This approach ensures feature comparability with queries (also 25-30 words).
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np

import spacy
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceLevelFeatureExtractor:
    """Extract features from sentences of specific length (25-30 words)."""

    def __init__(self, nlp_model: str = "en_core_web_sm"):
        """Initialize spaCy model."""
        try:
            self.nlp = spacy.load(nlp_model)
            logger.info(f"Loaded spaCy model: {nlp_model}")
        except OSError:
            logger.error(f"spaCy model {nlp_model} not found. Install with: python -m spacy download {nlp_model}")
            raise

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences

    def count_words(self, text: str) -> int:
        """Count words in text (simple whitespace-based count)."""
        # Remove extra whitespace and split
        words = re.findall(r'\b\w+\b', text.lower())
        return len(words)

    def extract_profilingud_features(self, text: str) -> Dict[str, Any]:
        """
        Extract ProfilingUD-style features from text.

        Compatible with BatchLocalFeatureExtractor output format.
        """
        doc = self.nlp(text)

        if len(doc) == 0:
            return {}

        n_tokens = len(doc)
        n_sentences = len(list(doc.sents))

        if n_sentences == 0:
            n_sentences = 1

        # Basic statistics
        tokens_per_sent = n_tokens / n_sentences
        char_per_tok = sum(len(token.text) for token in doc) / n_tokens if n_tokens > 0 else 0

        # UPOS distribution
        upos_counts = defaultdict(int)
        for token in doc:
            if token.pos_:
                upos_counts[f"upos_dist_{token.pos_}"] += 1

        upos_dist = {k: (v / n_tokens) * 100 for k, v in upos_counts.items()}

        # TTR (Type-Token Ratio)
        lemmas = [token.lemma_.lower() for token in doc if token.lemma_ and token.is_alpha]
        forms = [token.text.lower() for token in doc if token.is_alpha]

        ttr_lemma_chunks_100 = len(set(lemmas)) / len(lemmas) if lemmas else 0
        ttr_form_chunks_100 = len(set(forms)) / len(forms) if forms else 0
        ttr_lemma_chunks_200 = ttr_lemma_chunks_100  # Simplified
        ttr_form_chunks_200 = ttr_form_chunks_100  # Simplified

        # Lexical density (content words / total words)
        content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'}
        content_words = sum(1 for token in doc if token.pos_ in content_pos)
        lexical_density = content_words / n_tokens if n_tokens > 0 else 0

        # Verb tense distribution
        verbs = [token for token in doc if token.pos_ == "VERB" and token.morph.get("Tense")]

        if verbs:
            past_count = sum(1 for v in verbs if "Past" in v.morph.get("Tense"))
            pres_count = sum(1 for v in verbs if "Pres" in v.morph.get("Tense"))
            total_tensed = past_count + pres_count

            if total_tensed > 0:
                verbs_tense_dist_Past = (past_count / total_tensed) * 100
                verbs_tense_dist_Pres = (pres_count / total_tensed) * 100
            else:
                verbs_tense_dist_Past = 0.0
                verbs_tense_dist_Pres = 0.0
        else:
            verbs_tense_dist_Past = 0.0
            verbs_tense_dist_Pres = 0.0

        # Verb mood distribution
        verb_moods = [token for token in doc if token.pos_ == "VERB" and token.morph.get("Mood")]

        if verb_moods:
            imp_count = sum(1 for v in verb_moods if "Imp" in v.morph.get("Mood"))
            ind_count = sum(1 for v in verb_moods if "Ind" in v.morph.get("Mood"))
            total_mooded = imp_count + ind_count

            if total_mooded > 0:
                verbs_mood_dist_Imp = (imp_count / total_mooded) * 100
                verbs_mood_dist_Ind = (ind_count / total_mooded) * 100
            else:
                verbs_mood_dist_Imp = 0.0
                verbs_mood_dist_Ind = 100.0
        else:
            verbs_mood_dist_Imp = 0.0
            verbs_mood_dist_Ind = 100.0

        # Verb form distribution
        verb_forms = defaultdict(int)
        for token in doc:
            if token.pos_ == "VERB" and token.tag_:
                form_map = {
                    "VBZ": "Fin", "VBD": "Fin", "VBP": "Fin",
                    "VBG": "Ger", "VB": "Inf", "VBN": "Part"
                }
                if token.tag_ in form_map:
                    verb_forms[f"verbs_form_dist_{form_map[token.tag_]}"] += 1

        total_verb_forms = sum(verb_forms.values())
        if total_verb_forms > 0:
            verbs_form_dist = {k: (v / total_verb_forms) * 100 for k, v in verb_forms.items()}
        else:
            verbs_form_dist = {
                "verbs_form_dist_Fin": 50.0,
                "verbs_form_dist_Ger": 12.5,
                "verbs_form_dist_Inf": 25.0,
                "verbs_form_dist_Part": 12.5
            }

        # Subject/Object position (relative to verb)
        subj_pre = subj_post = obj_pre = obj_post = 0
        total_subj = total_obj = 0

        for token in doc:
            if token.dep_ in ("nsubj", "nsubj:pass"):
                total_subj += 1
                # Find verb head
                head = token.head
                if token.i < head.i:
                    subj_pre += 1
                else:
                    subj_post += 1
            elif token.dep_ == "obj":
                total_obj += 1
                head = token.head
                if token.i < head.i:
                    obj_pre += 1
                else:
                    obj_post += 1

        if total_subj > 0:
            subj_pre = (subj_pre / total_subj) * 100
            subj_post = (subj_post / total_subj) * 100
        else:
            subj_pre = 50.0
            subj_post = 50.0

        if total_obj > 0:
            obj_pre = (obj_pre / total_obj) * 100
            obj_post = (obj_post / total_obj) * 100
        else:
            obj_pre = 50.0
            obj_post = 50.0

        # Combine all features
        features = {
            "n_sentences": float(n_sentences),
            "n_tokens": float(n_tokens),
            "tokens_per_sent": tokens_per_sent,
            "char_per_tok": char_per_tok,
            "ttr_lemma_chunks_100": ttr_lemma_chunks_100,
            "ttr_lemma_chunks_200": ttr_lemma_chunks_200,
            "ttr_form_chunks_100": ttr_form_chunks_100,
            "ttr_form_chunks_200": ttr_form_chunks_200,
            "lexical_density": lexical_density * 100,  # Convert to percentage
            "verbs_tense_dist_Past": verbs_tense_dist_Past,
            "verbs_tense_dist_Pres": verbs_tense_dist_Pres,
            "verbs_mood_dist_Imp": verbs_mood_dist_Imp,
            "verbs_mood_dist_Ind": verbs_mood_dist_Ind,
            **upos_dist,
            **verbs_form_dist,
            "subj_pre": subj_pre,
            "subj_post": subj_post,
            "obj_pre": obj_pre,
            "obj_post": obj_post,
        }

        return features

    def extract_user_features_from_reviews(
        self,
        reviews: List[Dict[str, Any]],
        min_words: int = 25,
        max_words: int = 30,
        min_sentences: int = 5
    ) -> Dict[str, float]:
        """
        Extract averaged features from sentences of specific length.

        Args:
            reviews: List of review dicts with 'review_text' field
            min_words: Minimum sentence length
            max_words: Maximum sentence length
            min_sentences: Minimum number of qualifying sentences required

        Returns:
            Dictionary of averaged features
        """
        all_sentences = []

        # Extract all sentences from all reviews
        for review in reviews:
            review_text = review.get("review_text", "")
            if not review_text:
                continue

            sentences = self.split_into_sentences(review_text)

            for sent in sentences:
                word_count = self.count_words(sent)
                if min_words <= word_count <= max_words:
                    all_sentences.append(sent)

        if len(all_sentences) < min_sentences:
            logger.warning(f"Only found {len(all_sentences)} sentences in range [{min_words}, {max_words}]")
            # Still proceed with available sentences

        if not all_sentences:
            return {}

        logger.info(f"Extracting features from {len(all_sentences)} sentences (25-30 words)")

        # Extract features from each sentence
        all_features = []
        for sentence in tqdm(all_sentences, desc="Extracting sentence features"):
            features = self.extract_profilingud_features(sentence)
            if features:
                all_features.append(features)

        if not all_features:
            return {}

        # Average features across all sentences
        averaged_features = self._average_features(all_features)

        # Add metadata
        averaged_features["_metadata"] = {
            "n_sentences_used": len(all_sentences),
            "min_word_count": min_words,
            "max_word_count": max_words
        }

        return averaged_features

    def _average_features(self, feature_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Average features across multiple feature dictionaries."""
        if not feature_list:
            return {}

        # Collect all feature keys
        all_keys = set()
        for features in feature_list:
            all_keys.update(features.keys())

        # Calculate average for each key
        averaged = {}
        for key in all_keys:
            values = [f.get(key, 0) for f in feature_list if key in f]
            if values:
                # Filter out non-numeric values
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    averaged[key] = float(np.mean(numeric_values))

        return averaged


def main():
    """Main extraction workflow."""
    # Paths
    review_data_path = Path("/home/wlia0047/wenyu/data/Amazon-Reviews-2018/processed/user_reviews/user_product_reviews.json")
    output_dir = Path("/home/wlia0047/wenyu/result/user_profile/08_multi_candidate_filtering/sentence_features")
    dual_queries_dir = Path("/home/wlia0047/wenyu/result/user_profile/06_query")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get target users from dual_queries directory
    target_users = set()
    for f in dual_queries_dir.glob("dual_queries_*.json"):
        user_id = f.stem.replace("dual_queries_", "")
        target_users.add(user_id)

    logger.info(f"Found {len(target_users)} target users from dual_queries")

    if not target_users:
        logger.warning("No target users found in dual_queries directory!")
        return

    # Load review data
    logger.info(f"Loading review data from {review_data_path}")
    with open(review_data_path, 'r') as f:
        user_reviews = json.load(f)

    # Initialize extractor
    extractor = SentenceLevelFeatureExtractor()

    # Process only target users
    results = {}
    for user_id in sorted(target_users):
        if user_id not in user_reviews:
            logger.warning(f"User {user_id} not found in review data")
            continue

        user_data = user_reviews[user_id]
        reviews = user_data.get("reviews", [])

        if not reviews:
            logger.warning(f"No reviews found for user {user_id}")
            continue

        logger.info(f"Processing user {user_id} ({len(reviews)} reviews)")

        # Extract features from 25-30 word sentences
        features = extractor.extract_user_features_from_reviews(
            reviews,
            min_words=25,
            max_words=30,
            min_sentences=3  # Allow users with fewer qualifying sentences
        )

        if features:
            metadata = features.pop("_metadata", {})

            results[user_id] = {
                "user_id": user_id,
                "profilingud_features": features,
                "metadata": metadata
            }
            logger.info(f"Successfully extracted features for {user_id}")
        else:
            logger.warning(f"Failed to extract features for {user_id}")

    # Save results
    output_path = output_dir / "sentence_level_features_all_users.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved features for {len(results)} users to {output_path}")

    # Also save individual files for compatibility
    for user_id, user_data in results.items():
        individual_path = output_dir / f"sentence_level_features_{user_id}.json"
        with open(individual_path, 'w') as f:
            json.dump(user_data, f, indent=2)

    logger.info(f"Saved individual feature files to {output_dir}")


if __name__ == "__main__":
    main()
