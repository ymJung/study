import requests
from flask import Flask, request, jsonify
import configparser
import json
import logging
from logging.handlers import TimedRotatingFileHandler

config = configparser.ConfigParser()
config.read('config.cfg', encoding='utf-8')

GITLAB_TOKEN = config['gitlab']['API_KEY']
SLACK_URL = config['bfb']['slack_url']

OLLAMA_URL = "http://127.0.0.1:11434"
GITLAB_API_URL = config['gitlab']['API_URL']
CHANNEL_URL = config['bfb']['channel']
SRC_PATH = 'src/main'


def get_branch_diff(project_id, source_branch, target_branch):
    headers = {
        "Private-Token": GITLAB_TOKEN
    }
    compare_url = f"{GITLAB_API_URL}/projects/{project_id}/repository/compare"
    params = {
        "from": target_branch,
        "to": source_branch
    }
    response = requests.get(compare_url, headers=headers, params=params)
    if response.status_code == 200:
        diffs = response.json().get("diffs", [])
        diff_strings = []
        for diff in diffs:
            new_path = diff.get('new_path', 'unknown')
            old_path = diff.get('old_path', 'unknown')
            diff_content = diff.get('diff', '')
            diff_string = f"New path: {new_path}\nOld path: {old_path}\nChanges:\n{diff_content}"
            diff_strings.append(diff_string)        
        return "\n\n".join(diff_strings)
    return f"Error: {response.status_code}, {response.text}"

def conversation(payload):    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(f"{OLLAMA_URL}/api/chat", headers=headers, data=json.dumps(payload), stream=True)
    return response

def response_to_json(response):
    if response.status_code == 200:
        collected_content = ''
        for line in response.iter_lines():
            if line:
                try:                    
                    result = json.loads(line.decode('utf-8'))
                    message = result.get("message", {})
                    content = message.get("content", "")
                    collected_content += content
                    if result.get("done", False):
                        break  # Exit the loop when done is True
                except json.JSONDecodeError:
                    continue  # Skip lines that aren't valid JSON
        return collected_content.strip()
    else:
        print(f"HTTP Error: {response.status_code}, Response: {response.text}")
        return f"Error: {response.status_code}, {response.text}"

def get_code_review(diff_msg):
    response = conversation({
        "model": "qwen2.5",
        "temperature": 0.7,  # CLI 기본값
        "max_tokens": 4096,  # 모델의 기본 최대 토큰 수
        "top_p": 1.0,        # 기본값
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant for reviewing code changes in a GitLab Merge Request (MR). Provide concise feedback. answer language is Korean.(한글)"
            },
            {
                "role": "user",
                "content": f"""
Analyze the following git diff output and provide feedback focusing on:
This project API server is built with Java, Spring boot, Mybatis, Kafka, AWS cloud.
1. Critical logical issues.
2. Key style improvements.
Use this format:
- **Issue**: [Short description]
- **Suggestion**: [Short suggestion]
Here is the git diff:
{diff_msg}
"""
            }
        ]
    })
    return response_to_json(response)
    


def send_slack(source, target, origin_merge_url, review_result):
    msg = f'auto generated diff review message \n source [{source}] target[{target}] \n'
    if origin_merge_url and len(origin_merge_url) > 0 : msg += f"branch 에 대한 MR-review 입니다.\n{origin_merge_url}\n"
    msg += review_result    
    app.logger.info(f'review msg: \n {msg}')
    requests.post(SLACK_URL, json={'channel':CHANNEL_URL, 'text':msg}, headers={'Content-Type':'application/json'})


def setup_logger(app):
    app.logger.setLevel(logging.INFO) 
    file_handler = TimedRotatingFileHandler(
        "flask_llm.log", when="midnight", interval=1, backupCount=7  # Rotate daily, keep 7 days
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)


## start ##    
app = Flask(__name__)
setup_logger(app)

@app.route('/review', methods=['GET'])
def review():
    project_id = request.args.get('projectId')
    source = request.args.get('from')
    target = request.args.get('to')
    app.logger.info(f'request review : {project_id}, {source}, {target}')
    if target not in ['develop', 'master']:
        return jsonify({"review": 'ok'})
    origin_merge_url = request.args.get('originMergeUrl')  
    diff = get_branch_diff(project_id, source, target)
    review_text = get_code_review(diff)
    if review_text : send_slack(source, target, origin_merge_url, review_text)

    return jsonify({"review": review_text})

@app.route('/prompt', methods=['GET'])
def prompt():
    text = request.args.get('text')
    response = conversation({
        "model": "qwen2.5",
        "temperature": 0.7,  # CLI 기본값
        "max_tokens": 4096,  # 모델의 기본 최대 토큰 수
        "top_p": 1.0,        # 기본값
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "messages": [
            {
                "role": "system",
                "content": "kind assist."
            },
            {
                "role": "user",
                "content": f"""
{text}
"""
            }
        ]
    })

    return jsonify({"review": response_to_json(response)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=15000, debug=True)