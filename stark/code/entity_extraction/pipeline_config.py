import os

# Configuration Constants
TARGET_USER = "AG7EF0SVBQOUX"  # The ID of the target user

# Paths
def get_workspace_root():
    # Helper to get the workspace root directory
    # Current: stark/code/entity_extraction/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up: entity_extraction -> code -> stark -> wenyu (root)
    return os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

WORKSPACE_ROOT = get_workspace_root()
RESULT_DIR = os.path.join(WORKSPACE_ROOT, "result")

# Result Files
PRODUCT_ENTITIES_FILE = os.path.join(RESULT_DIR, "product_entities.json")
PRODUCT_ENTITIES_NORMALIZED_FILE = os.path.join(RESULT_DIR, "product_entities_normalized.json")
PRODUCT_ENTITIES_WITH_DIMS_FILE = os.path.join(RESULT_DIR, "product_entities_with_std_dimensions.json")
USER_PREFERENCES_FILE = os.path.join(RESULT_DIR, "user_preference_entities.json")
MATCHED_ENTITIES_FILE = os.path.join(RESULT_DIR, "entity_matching_results.json")
MATCHED_ENTITIES_NORMALIZED_FILE = os.path.join(RESULT_DIR, "entity_matching_results_normalized.json")
API_RESPONSES_FILE = os.path.join(RESULT_DIR, "main_api_raw_responses.json")

# Ensure result directory exists
os.makedirs(RESULT_DIR, exist_ok=True)
