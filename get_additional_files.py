from huggingface_hub import hf_hub_download, HfFileSystem

def get_additional_files():
    try:
        repo_id = "wanndev14/yolo-nutrition-api"
        files_to_read = ['fix_yolo_compatibility.py', 'api.py', 'Dockerfile']
        
        for file in files_to_read:
            try:
                file_path = hf_hub_download(repo_id=repo_id, filename=file, repo_type="space")
                print(f"\n--- Content of {file} ---")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(content)
                print(f"--- End of {file} ---\n")
            except Exception as e:
                print(f"Could not read {file}: {e}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_additional_files()