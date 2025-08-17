# create_zip.py (v2)
import os
import zipfile

print("--- Creating Correctly Named TabNet Zip Archive ---")

# --- Configuration ---
# The directory where the original model files are
model_dir = '/home/abhijit-42/NNDL/proj-final/models/tabnet/'

# The names of your SOURCE files
source_network_file = 'tabnet_network.pt'
source_params_file = 'tabnet_model_params.json'

# The base name for our new zip file (e.g., 'tabnet_model')
zip_basename = 'tabnet_model'
# --- End of Configuration ---


# --- File Paths ---
zip_filename = f"{zip_basename}.zip"
zip_path = os.path.join(model_dir, zip_filename)

# These are the required names INSIDE the zip file
target_network_arcname = f"{zip_basename}_network.pt"
target_params_arcname = f"{zip_basename}_model_params.json"

# Full paths to the original source files
source_network_path = os.path.join(model_dir, source_network_file)
source_params_path = os.path.join(model_dir, source_params_file)


# --- Main Logic ---
if not os.path.exists(source_network_path) or not os.path.exists(source_params_path):
    print(f"❌ ERROR: Cannot find source files in '{model_dir}'.")
else:
    try:
        print(f"Creating '{zip_filename}'...")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add the source files to the zip, but give them the target names
            print(f"  -> Adding '{source_network_file}' as '{target_network_arcname}'")
            zipf.write(source_network_path, arcname=target_network_arcname)
            
            print(f"  -> Adding '{source_params_file}' as '{target_params_arcname}'")
            zipf.write(source_params_path, arcname=target_params_arcname)
        
        print(f"\n🎉 SUCCESS: Created '{zip_filename}' with correctly named internal files.")
        print("Your streamlit app should now work without any changes.")

    except Exception as e:
        print(f"\n❌ FAILED: Could not create the zip file. Error: {e}")

print("\n--- Script Finished ---")