import json
import openai
import configparser
config = configparser.ConfigParser()
config.read('config.cfg')
openai.api_key = config['openai']['TOKEN']


"""
MCP 스터디
* MCP ?? 질문을 LLM이 판단하여 API를 호출한다
* 내가만든건?? 질문을 LLM이 판단하여 가지고 있는 API의 경우 호출한다. (ex.일기예보)
"""
# 예시 질문:
# 1. "서울 날씨 알려줘"
# 2. "부산 날씨 어때?"
# 3. "오늘 날씨가 어떤가요?"

# 예시 function (동작안함-필요시 API구현)
def get_weather(location):
    if location.lower() == "seoul":
        return {"temp": "12°C", "condition": "Clear"}
    else:
        return {"temp": "unknown", "condition": "unknown"}

# ChatGPT API를 호출하는 함수
def call_chatgpt(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages
    )
    return response.choices[0].message.content.strip()

# 사용자 요청을 처리하여 MCP 대화 구조를 생성하는 함수
def process_conversation(user_request):
    conversation = {
        "actors": [
            {"id": "user", "type": "human"},
            {"id": "assistant", "type": "chatgpt"},
            {"id": "weather_api", "type": "tool"}
        ],
        "messages": []
    }
    
    # 1. 사용자의 자연어 요청 추가
    conversation["messages"].append({
        "actor_id": "user",
        "content": user_request
    })
    
    # LLM에 role 설정 
    context = [
        {"role": "system", "content": "당신은 API 요청을 처리하는 도우미입니다. 사용자의 요청을 적절히 이해하여 필요한 도구 호출이나 응답을 구성하세요."}
    ]
    
    # "날씨" 키워드가 있으면 weather_api 도구를 호출하는 흐름을 실행
    if "날씨" in user_request:
        # 2. assistant가 weather_api 호출 의도를 표현하는 function_call 메시지 추가
        conversation["messages"].append({
            "actor_id": "assistant",
            "function_call": {
                "name": "getWeather",
                "arguments": {"location": "Seoul"}  # 예시로 'Seoul' 고정 처리
            }
        })
        
        # 3. weather_api 도구 호출 및 결과 추가
        weather_result = get_weather("Seoul")
        conversation["messages"].append({
            "actor_id": "weather_api",
            "content": json.dumps(weather_result, ensure_ascii=False)
        })
        
        # 4. ChatGPT를 호출하여 최종 사용자 응답 생성
        # 여기서는 weather_api 결과를 context에 추가하고 ChatGPT가 자연어 응답을 생성하도록 함
        context.append({"role": "assistant", "content": f"weather_result: {json.dumps(weather_result, ensure_ascii=False)}"})
        context.append({"role": "user", "content": "위 날씨 정보를 바탕으로 사용자에게 응답해줘."})
        chatgpt_response = call_chatgpt(context)
        
        conversation["messages"].append({
            "actor_id": "assistant",
            "content": chatgpt_response
        })
    else:
        # 날씨 이외의 요청은 바로 ChatGPT에 전달하여 응답 받음
        context.append({"role": "user", "content": user_request})
        chatgpt_response = call_chatgpt(context)
        conversation["messages"].append({
            "actor_id": "assistant",
            "content": chatgpt_response
        })
    
    return conversation

# 무한루프로 자연어 요청을 계속 받아 처리
while True:
    user_input = input("자연어 요청을 입력하세요 (종료하려면 q): ")
    if user_input.lower() in ["q"]:
        print("프로그램을 종료합니다.")
        break
    conv = process_conversation(user_input)
    print(json.dumps(conv, indent=4, ensure_ascii=False))