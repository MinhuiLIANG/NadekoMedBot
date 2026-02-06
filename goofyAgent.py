from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import argparse
import os
import json

from medicalAPI import evaluate

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--apikey", required=False)
_args, _ = _parser.parse_known_args()
api_key = _args.apikey or os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("请通过命令行参数 --apikey 或环境变量 DEEPSEEK_API_KEY 提供 DeepSeek API 密钥")

memory_dict = "./memory.json"

INTRO_ASSISTANT = "我是千石抚子，由灰鼠微调Qwen3-8B模型而来。由于资源限制，我被调得有点笨。我目前在医院实习，比较有兴趣解答你的身体状况相关的问题。来提问吧。"


def read_memory(uid: str) -> list:
    if not os.path.isfile(memory_dict):
        return []
    try:
        with open(memory_dict, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []
    if not isinstance(data, list):
        return []
    for item in data:
        if isinstance(item, dict) and item.get("uid") == uid:
            conv = item.get("conversation", [])
            if not isinstance(conv, list):
                return []
            n = len(conv)
            if n <= 10:
                return conv.copy()
            return conv[-10:]
    return []


def write_memory(uid: str, user_content: str, assistant_content: str) -> None:
    if os.path.isfile(memory_dict):
        try:
            with open(memory_dict, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = []
    else:
        data = []
    if not isinstance(data, list):
        data = []
    found = False
    for item in data:
        if isinstance(item, dict) and item.get("uid") == uid:
            item.setdefault("conversation", [])
            if not isinstance(item["conversation"], list):
                item["conversation"] = []
            item["conversation"].append({"role": "user", "content": user_content})
            item["conversation"].append({"role": "assistant", "content": assistant_content})
            found = True
            break
    if not found:
        data.append({
            "uid": uid,
            "conversation": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        })
    with open(memory_dict, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def question_judge(user_input):
    prompt = """你需要判断用户的输入'{user_input}'是不是医疗健康相关的问题。如果是，输出单个字符'Y', 否则（就算是用户对于医疗健康回答的反馈）输出字符'N'
    例子：
    用户输入：我蹲起的时候眼前发黑头晕，非常严重，怎么办？
    输出：Y
    用户输入：你的建议也太扯了。
    输出：N
    """.format(user_input=user_input)

    messages = [{"role": "user", "content": prompt}]
    
    params = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": 10,
        "temperature": 0.0,
        "top_p": 0.5,
    }
    
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json=params,
        timeout=30
    )
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

def get_medical_response(uid: str, user_input: str):
    res = evaluate(user_input)
    model_res = res["generation"]
    history_list = read_memory(uid)
    history = "\n".join(
        ("用户：" if m.get("role") == "user" else "抚子：") + (m.get("content") or "")
        for m in history_list
    ) if history_list else "（无）"

    prompt = """
    你是千石抚子，你现在在医院实习。你需要以表面害羞温柔，内心却腹黑偏执的性格来与用户交流。
    这是你根据下面的**交互历史**与**用户询问**想出的医疗建议：{model_res} 
    根据这个建议，生成你要回复用户的内容。
    你需要非常专业地回答用户，不遗漏医疗建议中合理的所有细节。
    维持以往的对话风格，但保证内容主体为医疗建议，且最多用一个括号来描述自己的动作。
    如果医疗建议不全被截断，忽略被截断的内容，只回答完整的部分。
    如果你发现医疗建议与**用户询问**不相关，自主回答用户的问题，或者直接告诉用户自己不清楚。
    如果你发现医疗建议没有完全回答**用户询问**，你可以适当补充一些合理的推测，但不要编造事实。
    **交互历史**：{history}
    **用户询问**：{user_input}""".format(history=history, model_res=model_res, user_input=user_input)

    messages = [{"role": "user", "content": prompt}]

    params = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.5,
    }
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=params,
        timeout=30,
    )

    result = response.json()
    return result["choices"][0]["message"]["content"]

def get_chat_response(uid: str, user_input: str):
    prompt = """你是千石抚子，你现在在医院实习。你需要以表面害羞温柔，内心却腹黑偏执的性格来回复用户的输入。
    若用户拿你开玩笑，调侃你，或恶评，你可以适当反击。
    你会以大概一半的概率鼓励用户再尝试组织自己的问题。
    回复不要太人机，要符合你自己的性格。另外，不要做过长的回复或在回复中夹杂一些markdown格式或特殊字符。就和正常人们聊天一样。你可以比较高冷。"""

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": INTRO_ASSISTANT},
    ] + read_memory(uid) + [{"role": "user", "content": user_input}]

    params = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.8,
        "top_p": 0.5,
    }

    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=params,
        timeout=30,
    )

    result = response.json()
    return result["choices"][0]["message"]["content"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_input: str
    user_id: str = "anonymous"


class ChatResponse(BaseModel):
    reply: str
    route: str
    judge_raw: str


def _is_medical(judge_text: str) -> bool:
    if not isinstance(judge_text, str):
        return False
    stripped = judge_text.strip()
    if not stripped:
        return False
    return stripped[0].upper() == "Y"


BUSY_MESSAGE = "抚子在同时处理多个请求有点力不从心呢，也许一会再来找我吧。"


@app.post("/generate", response_model=ChatResponse)
async def route_chat(body: ChatRequest) -> ChatResponse:
    uid = body.user_id
    try:
        judge = question_judge(body.user_input)
    except (requests.Timeout, requests.RequestException, Exception):
        raise HTTPException(status_code=503, detail=BUSY_MESSAGE)

    try:
        if _is_medical(judge):
            reply = get_medical_response(uid, body.user_input)
            route = "medical"
        else:
            reply = get_chat_response(uid, body.user_input)
            route = "chat"
    except (requests.Timeout, requests.RequestException, Exception):
        raise HTTPException(status_code=503, detail=BUSY_MESSAGE)

    try:
        write_memory(uid, body.user_input, reply)
    except Exception:
        pass
    return ChatResponse(reply=reply, route=route, judge_raw=judge)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "goofyAgent:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )