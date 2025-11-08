import os
from huggingface_hub import HfApi, Repository
import tempfile
import shutil

def analyze_hf_space():
    try:
        # Initialize HF API
        api = HfApi()
        
        # Space info
        repo_id = "wanndev14/yolo-nutrition-api"
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Downloading files from {repo_id}...")
            
            # Clone the space
            repo = Repository(local_dir=temp_dir, 
                            clone_from=repo_id,
                            repo_type="space")
            
            print(f"Files downloaded to: {temp_dir}")
            
            # List all files
            print("\nFiles in the space:")
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, temp_dir)
                    print(f"  {rel_path}")
                    
                    # Read and display key files
                    if file in ['app.py', 'requirements.txt', 'README.md']:
                        print(f"\n--- Content of {rel_path} ---")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            print(content)
                        print(f"--- End of {rel_path} ---\n")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Trying alternative approach...")
        
        # Alternative: try to get file list directly
        try:
            repo_files = api.list_repo_files(repo_id=repo_id, repo_type="space")
            print(f"Files in {repo_id}:")
            for file in repo_files:
                print(f"  {file}")
                
                # Read specific files
                if file in ['app.py', 'requirements.txt', 'README.md']:
                    try:
                        content = api.read_file(repo_id=repo_id, filename=file, repo_type="space")
                        print(f"\n--- Content of {file} ---")
                        print(content.decode('utf-8'))
                        print(f"--- End of {file} ---\n")
                    except Exception as e2:
                        print(f"Could not read {file}: {e2}")
                        
        except Exception as e2:
            print(f"Alternative approach also failed: {e2}")

if __name__ == "__main__":
    analyze_hf_space()