# test_load.py
import os
from pytorch_tabnet.tab_model import TabNetClassifier

print("--- Starting TabNet Load Test ---")

# Define the absolute path directly from your error message.
model_path_prefix = '/home/abhijit-42/NNDL/proj-final/models/tabnet/tabnet'

print(f"Attempting to load model with prefix: {model_path_prefix}")

# --- Extra Debugging Checks ---
# Let's verify if Python can see the component files using this prefix.
expected_pt_file = model_path_prefix + '_network.pt'
expected_json_file = model_path_prefix + '_model_params.json'

print(f"Checking for file: {expected_pt_file}")
if os.path.exists(expected_pt_file):
    print("    ✅ Found _network.pt file.")
else:
    print("    ❌ CRITICAL: Could NOT find _network.pt file at this path.")

print(f"Checking for file: {expected_json_file}")
if os.path.exists(expected_json_file):
    print("    ✅ Found _model_params.json file.")
else:
    print("    ❌ CRITICAL: Could NOT find _model_params.json file at this path.")
# --- End of Checks ---

try:
    # Initialize a new TabNet classifier
    tabnet_model = TabNetClassifier()

    # Attempt to load the model
    tabnet_model.load_model(model_path_prefix)

    print("\n🎉 SUCCESS: Model loaded successfully! 🎉")

except Exception as e:
    print(f"\n❌ FAILED: The model could not be loaded.")
    print(f"    Error type: {type(e).__name__}")
    print(f"    Error message: {e}")

print("\n--- Test Finished ---")