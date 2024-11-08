import os
import json

def collect_java_files(directory_path):
    java_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    return java_files

def truncate_content(content, max_length=5000):
    if len(content) > max_length:
        return content[:max_length]  
    return content

def save_as_json_from_parent_directory(parent_directory, output_file_path, max_length=5000):
    code_files = []
    for project_directory in os.listdir(parent_directory):
        full_project_path = os.path.join(parent_directory, project_directory)
        if os.path.isdir(full_project_path):
            project_name = os.path.basename(os.path.normpath(full_project_path))
            java_files = collect_java_files(full_project_path)
            for file_path in java_files:
                with open(file_path, 'r') as file:
                    content = file.read()
                truncated_content = truncate_content(content, max_length=max_length)
                if len(truncated_content) > max_length:
                    print(f"Skipping file {file_path} as it exceeds max length after truncation.")
                    print(f"Content length: {len(truncated_content)}")
                    print("Content preview:")
                    print(truncated_content[:1000])  
                    continue

                code_files.append({
                    'project_name': project_name,
                    'file_name': os.path.basename(file_path),
                    'file_path': file_path,
                    'module': os.path.basename(os.path.dirname(file_path)),  
                    'content': truncated_content
                })

    with open(output_file_path, 'w') as output_file:
        json.dump(code_files, output_file, indent=4)
project_path = "/home/zero00/Dev/dataset/train"
output_path = "/home/zero00/Dev/datasets/output/multi_project_dataset.json"


save_as_json_from_parent_directory(project_path, output_path, max_length=5000)

print(f"Dataset saved in : {output_path}")

