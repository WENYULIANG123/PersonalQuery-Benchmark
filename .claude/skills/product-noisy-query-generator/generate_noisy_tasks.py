import json
import csv
import re
import argparse
import sys
import os

# --- Risk Logic (Using Unified Error Statistics) --

# Try to import spacy, handle if missing
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

class RiskWeightedInjector:
    def __init__(self, stats_file=None):
        """
        Initialize with error statistics from writing analysis.

        Args:
            stats_file: Path to error_stats_{USER_ID}.json file
        """
        # Default Weights (if no stats file provided)
        # Expanded with more common user error patterns
        self.lexical_map = {
            # Common spelling errors (replace actions - realistic errors)
            "will":       {"coef": 0.40, "action": "replace", "value": "wii"},
            "really":     {"coef": 0.35, "action": "replace", "value": "realley"},
            "definitely": {"coef": 0.40, "action": "replace", "value": "definitly"},
            "beautiful":  {"coef": 0.35, "action": "replace", "value": "beatiful"},
            "separate":   {"coef": 0.35, "action": "replace", "value": "seperate"},
            "received":   {"coef": 0.35, "action": "replace", "value": "recieved"},
            "until":      {"coef": 0.35, "action": "replace", "value": "untill"},
            "beginning":  {"coef": 0.35, "action": "replace", "value": "begining"},
            "necessary":  {"coef": 0.35, "action": "replace", "value": "neccesary"},
            "intact":     {"coef": 0.50, "action": "replace", "value": "in tact"},
            "fuchsia":    {"coef": 0.50, "action": "replace", "value": "fuschia"},
            "palette":    {"coef": 0.50, "action": "replace", "value": "pallet"},
        }

        # Grammar weights based on error categories
        self.grammar_weights = {
            "agreement": 0.30,        # Agreement errors (subject-verb, number consistency)
            "preposition": 0.30,      # Preposition errors (missing/wrong preposition)
            "suffix": 0.30,           # Suffix errors (word form, comparative)
            "hyphenation": 0.30,      # Hyphenation errors (missing hyphens)
            "collocation": 0.30,      # Collocation errors (unnatural word pairing)
        }

        # Structural weights (static)
        self.structure_weights = {
            "length_penalty": 0.15,
            "drafting_speed": 0.10
        }

        self.GLOBAL_THRESHOLD = 0.5  # Lowered from 0.7 to allow more modifications

        # Store error rate for dynamic threshold adjustment
        self.error_rate = 0.0

        # Load error statistics if provided
        if stats_file:
            self._load_stats(stats_file)

    def _load_stats(self, stats_path):
        """
        Load error statistics and update weights dynamically based on error counts.

        Weight calculation formula:
        - Base weight = 0.3
        - Added weight = (error_count / max_errors_in_category) * 0.6
        - Max weight = 0.95

        This ensures that frequently occurring error types get higher weights.
        """
        print(f"Loading error statistics from {stats_path}...")

        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)

            total_errors = stats.get('total_errors', 1)

            # --- 1. Process Spelling Errors ---
            spelling_stats = stats.get('spelling', {})
            spelling_total = stats.get('spelling_total', 1)

            print("\n📝 Spelling Error Weights:")
            for err_type, count in spelling_stats.items():
                # Calculate weight based on frequency
                weight = 0.3 + (count / spelling_total) * 0.6
                weight = min(weight, 0.95)
                print(f"  - {err_type}: {count} errors → weight = {weight:.2f}")

                # Map spelling error types to specific words/patterns
                self._map_spelling_error(err_type, count, weight, spelling_total)

            # --- 2. Process Grammar Errors ---
            grammar_stats = stats.get('grammar', {})
            grammar_total = stats.get('grammar_total', 1)

            print("\n📐 Grammar Error Weights:")
            for err_type, count in grammar_stats.items():
                weight = 0.3 + (count / grammar_total) * 0.6
                weight = min(weight, 0.95)
                print(f"  - {err_type}: {count} errors → weight = {weight:.2f}")

                # Map to internal grammar categories
                self._map_grammar_error(err_type, weight)

            print(f"\n✅ Statistics loaded: {spelling_total} spelling errors, {grammar_total} grammar errors")
            print(f"   Total error rate: {stats.get('errors_per_100_words', 0):.2f} per 100 words")

            # Store error rate and adjust threshold dynamically
            self.error_rate = stats.get('errors_per_100_words', 0)

            # Dynamic threshold adjustment based on error rate
            # Higher error rate = lower threshold (easier to trigger noise)
            # Lower error rate = higher threshold (harder to trigger noise)
            # Formula: threshold = 0.7 / (1 + error_rate)
            # Example:
            #   error_rate = 0.45 → threshold = 0.7 / 1.45 = 0.48
            #   error_rate = 1.50 → threshold = 0.7 / 2.50 = 0.28
            #   error_rate = 0.10 → threshold = 0.7 / 1.10 = 0.64
            original_threshold = self.GLOBAL_THRESHOLD
            self.GLOBAL_THRESHOLD = original_threshold / (1 + self.error_rate)

            print(f"   Original threshold: {original_threshold:.2f}")
            print(f"   Adjusted threshold: {self.GLOBAL_THRESHOLD:.2f} (based on error rate {self.error_rate:.2f})\n")

        except Exception as e:
            print(f"❌ Error loading statistics: {e}")

    def _map_spelling_error(self, err_type, count, weight, total):
        """Map spelling error types to lexical triggers."""

        # Extra Space: "in tact" -> "intact", "note book" -> "notebook"
        if err_type == "Extra Space":
            self.lexical_map["intact"] = {"coef": weight, "action": "replace", "value": "in tact"}

        # Homophone: "pallet" -> "palette", "there" -> "their"
        elif err_type == "Homophone":
            self.lexical_map["palette"] = {"coef": weight, "action": "replace", "value": "pallet"}
            self.lexical_map["their"] = {"coef": weight * 0.5, "action": "replace", "value": "there"}

        # Scramble: "invididual" -> "individual", "Albrect" -> "Albrecht"
        elif err_type == "Scramble":
            self.lexical_map["individual"] = {"coef": weight, "action": "replace", "value": "invididual"}
            self.lexical_map["albrecht"] = {"coef": weight, "action": "replace", "value": "albrect"}

        # Substitution: "Theses" -> "These"
        elif err_type == "Substitution":
            self.lexical_map["these"] = {"coef": weight * 0.5, "action": "replace", "value": "theses"}

        # Insertion: "obssessed" -> "obsessed"
        elif err_type == "Insertion":
            self.lexical_map["obsessed"] = {"coef": weight, "action": "replace", "value": "obssessed"}

        # Hard Word: "Fuschia" -> "Fuchsia", "Nancy Zeimen" -> "Nancy Zieman"
        elif err_type == "Hard Word":
            self.lexical_map["fuchsia"] = {"coef": weight, "action": "replace", "value": "fuschia"}
            self.lexical_map["zieman"] = {"coef": weight, "action": "replace", "value": "zeimen"}

    def _map_grammar_error(self, err_type, weight):
        """Map grammar error types to internal categories."""

        # Agreement: subject-verb, number consistency
        if err_type == "Agreement":
            self.grammar_weights["agreement"] = weight

        # Preposition: missing or wrong preposition
        elif err_type == "Preposition":
            self.grammar_weights["preposition"] = weight

        # Suffix: word form errors, comparative, verb tense
        elif err_type == "Suffix":
            self.grammar_weights["suffix"] = weight

        # Hyphenation: missing hyphens in compound adjectives
        elif err_type == "Hyphenation":
            self.grammar_weights["hyphenation"] = weight

        # Collocation: unnatural word pairing
        elif err_type == "Collocation":
            self.grammar_weights["collocation"] = weight

    def calculate_risk_score(self, text: str):
        """Calculate risk score and identify triggers for injecting noise."""
        total_score = 0.0
        triggers = []
        words = text.split()
        doc = nlp(text) if SPACY_AVAILABLE else None

        # --- Lexical Scanning (Spelling Errors) ---
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in self.lexical_map:
                data = self.lexical_map[clean_word]
                total_score += data['coef']
                triggers.append({
                    "type": "lexical",
                    "target": word,
                    "coef": data['coef'],
                    "action": data['action'],
                    "value": data.get('value')
                })

        # --- Grammatical Scanning ---

        # A. Agreement Errors (Subject-Verb)
        if SPACY_AVAILABLE and doc:
            for token in doc:
                if token.tag_ == "NNS":  # Plural noun
                    if token.head.pos_ == "VERB" or token.head.pos_ == "AUX":
                        verb = token.head.text.lower()
                        if verb in ['are', 'do', 'have']:
                            total_score += self.grammar_weights["agreement"]
                            triggers.append({
                                "type": "agreement_error",
                                "target": token.head.text,
                                "context": f"{token.text} {token.head.text}",
                                "coef": self.grammar_weights["agreement"]
                            })

        # B. Relative Clauses (Preposition/Suffix)
        # Pattern: which/that + is/are -> remove is/are
        rel_matches = list(re.finditer(r'\b(which|that|who)\s+(is|are)\b', text, re.IGNORECASE))
        if rel_matches:
            for m in rel_matches:
                total_score += self.grammar_weights["preposition"]
                triggers.append({
                    "type": "relative_copula_drop",
                    "target": m.group(0),
                    "coef": self.grammar_weights["preposition"]
                })

        # C. Conjunction Links (Collocation)
        # Pattern: because/but/so + pronoun + verb -> remove pronoun+verb
        conj_matches = list(re.finditer(r'\b(because|but|so)\s+(they|it|he|she|we)\s+(are|is|was|were)\b', text, re.IGNORECASE))
        if conj_matches:
            for m in conj_matches:
                total_score += self.grammar_weights["collocation"]
                triggers.append({
                    "type": "conjunction_drop",
                    "target": m.group(0),
                    "coef": self.grammar_weights["collocation"]
                })

        # D. Hyphenation (Remove hyphens)
        # Pattern: word-word -> word word
        hyp_matches = list(re.finditer(r'\b([a-z]+)-([a-z]+)\b', text, re.IGNORECASE))
        if hyp_matches:
            for m in hyp_matches:
                # Only apply if it's a common compound that should be hyphenated
                compound = m.group(0).lower()
                if any(hw in compound for hw in ['high', 'good', 'off', 'doodle', 'cross']):
                    total_score += self.grammar_weights["hyphenation"]
                    triggers.append({
                        "type": "hyphenation_error",
                        "target": m.group(0),
                        "coef": self.grammar_weights["hyphenation"]
                    })

        # --- Structural Scanning ---

        # Length penalty
        if len(words) > 15:
            total_score += self.structure_weights["length_penalty"]
            triggers.append({
                "type": "structure_length",
                "target": "Sentence Length > 15",
                "coef": self.structure_weights["length_penalty"]
            })

        # Common words boost
        common_words = {'the', 'and', 'is', 'it', 'to', 'of'}
        common_count = sum(1 for w in words if w.lower() in common_words)
        if common_count > 3:
            total_score += self.structure_weights["drafting_speed"]
            triggers.append({
                "type": "structure_speed",
                "target": "Many common function words",
                "coef": self.structure_weights["drafting_speed"]
            })

        return total_score, triggers

    def apply_rewrite(self, text: str, triggers: list) -> str:
        """
        Apply noise based on triggers.

        Strategy: Apply ONLY the single highest-weighted trigger.
        Each sentence gets at most ONE error.
        This ensures natural-looking output that matches user writing style.
        """
        # Sort by weight (descending)
        triggers.sort(key=lambda x: x['coef'], reverse=True)

        # Filter triggers: only consider those with weight >= 0.3
        significant_triggers = [t for t in triggers if t['coef'] >= 0.3]

        # If no significant triggers, return original
        if not significant_triggers:
            return text

        # Apply ONLY the highest-weighted trigger (single error per sentence)
        trigger = significant_triggers[0]
        modified_text = text

        t_type = trigger['type']
        if t_type == "lexical":
            target = trigger['target']
            action = trigger['action']
            # Only apply replace actions (avoid artificial repeats)
            if action == "replace" and trigger.get('value'):
                pattern = re.compile(re.escape(target), re.IGNORECASE)
                modified_text = pattern.sub(trigger['value'], modified_text, count=1)
            else:
                # Skip repeat actions - they create artificial patterns
                return text

        elif t_type == "agreement_error":
            verb = trigger['target']
            deg_map = {'are': 'is', 'do': 'does', 'have': 'has'}
            if verb.lower() in deg_map:
                bad_verb = deg_map[verb.lower()]
                modified_text = re.sub(r'\b' + verb + r'\b', bad_verb, modified_text, count=1)

        elif t_type in ["relative_copula_drop", "conjunction_drop"]:
            phrase = trigger['target']
            keep = phrase.split()[0]
            modified_text = modified_text.replace(phrase, keep, 1)

        elif t_type == "hyphenation_error":
            target = trigger['target']
            replacement = target.replace('-', ' ')
            modified_text = modified_text.replace(target, replacement, 1)

        return modified_text


def generate_tasks(input_csv, output_json, stats_file=None):
    """
    Generate noisy query tasks based on user error statistics.

    Args:
        input_csv: Path to clean_queries.csv
        output_json: Path to output task JSON
        stats_file: Path to error_stats_{USER_ID}.json
    """
    injector = RiskWeightedInjector(stats_file)
    tasks = []

    print(f"\n📂 Reading queries from {input_csv}...")

    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row['query']
            score, triggers = injector.calculate_risk_score(query)

            # Create prompt for agent
            if score >= injector.GLOBAL_THRESHOLD:
                status = "HIGH RISK"

                # Only apply rewrites for HIGH RISK queries
                suggestion = injector.apply_rewrite(query, triggers)

                instruction = (
                    f"Risk Score: {score:.2f} (>= {injector.GLOBAL_THRESHOLD:.2f}). This query is vulnerable!\n"
                    f"Triggers found: {len(triggers)}, {len([t for t in triggers if t['coef'] >= 0.3])} significant (weight >= 0.3).\n"
                    f"Suggestion: Adopt the machine suggested changes, or manually inject similar errors "
                    f"based on the user's error patterns."
                )
            else:
                status = "LOW RISK"
                # For LOW RISK, suggestion is the original query (no changes)
                suggestion = query

                instruction = (
                    f"Risk Score: {score:.2f} (< {injector.GLOBAL_THRESHOLD:.2f}). The query is relatively safe.\n"
                    f"Keep it mostly original, or add 1 minor change if you feel like it."
                )

            task = {
                "id": row['id'],
                "original_query": query,
                "answer_ids_source": row.get('answer_ids_source', ''),
                "difficulty_score": round(score, 2),
                "risk_status": status,
                "triggers_detail": triggers,
                "machine_suggestion": suggestion,
                "agent_prompt": instruction
            }
            tasks.append(task)

    print(f"\n✅ Generated {len(tasks)} tasks.")

    # Write to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"📁 Saved to {output_json}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate noisy query tasks based on user error statistics"
    )
    parser.add_argument("--input", required=True, help="Input CSV file with clean queries")
    parser.add_argument("--output", required=True, help="Output JSON file for tasks")
    parser.add_argument(
        "--stats",
        required=False,
        help="Path to error_stats_{USER_ID}.json file with error frequencies"
    )

    args = parser.parse_args()

    generate_tasks(args.input, args.output, args.stats)
