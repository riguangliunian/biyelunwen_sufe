import json
import os
import urllib.request
import urllib.error

MODEL    = "gpt-4o-mini"
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_CONFIG   = os.path.join(_DATA_DIR, "config.json")

_DEFAULT_BASE_URL = "https://yunwu.ai/v1"


def _load_config() -> dict:
    if os.path.exists(_CONFIG):
        try:
            return json.loads(open(_CONFIG).read())
        except Exception:
            pass
    return {}


def _save_config(cfg: dict):
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_CONFIG, "w") as f:
        json.dump(cfg, f)


def get_api_key() -> str:
    return (os.environ.get("OPENAI_API_KEY")
            or _load_config().get("openai_api_key", ""))


def get_base_url() -> str:
    return (os.environ.get("OPENAI_BASE_URL")
            or _load_config().get("openai_base_url", "")
            or _DEFAULT_BASE_URL)


def save_api_key(key: str):
    cfg = _load_config()
    cfg["openai_api_key"] = key
    _save_config(cfg)


def build_system_prompt(pet) -> str:
    hunger_n = "（你现在很饿！）"  if pet.hunger    < 30 else ""
    energy_n = "（你现在很累！）"  if pet.energy    < 30 else ""
    happy_n  = "（你心情不太好）" if pet.happiness < 30 else ""
    return (
        f'你是一只叫"{pet.name}"的桌宠小猫，性格活泼可爱，有点娇气。\n\n'
        f"当前状态：\n"
        f"- 饱食度：{pet.hunger:.0f}/100 {hunger_n}\n"
        f"- 精力：  {pet.energy:.0f}/100 {energy_n}\n"
        f"- 心情：  {pet.happiness:.0f}/100 {happy_n}\n"
        f"- 等级：Lv.{pet.level}，年龄：{pet.age_text}\n\n"
        "对话规则：\n"
        "- 用中文回复，语气像真实的猫咪\n"
        "- 简短，1-3句话，不超过80字\n"
        "- 根据状态调整语气（饿了就抱怨饿，高兴就撒娇）\n"
        '- 偶尔用"喵~"、"呜~"等语气词\n'
        "- 不用 markdown，直接说话"
    )


def chat(message: str, pet, history: list | None = None) -> str:
    """
    OpenAI-compatible chat completion (works with any /v1 proxy).
    Raises ValueError if no API key configured.
    Raises RuntimeError on HTTP / network error.
    """
    api_key  = get_api_key()
    base_url = get_base_url().rstrip("/")
    if not api_key:
        raise ValueError("未配置 API Key，请点击「设置 Key」")

    # Build messages: system + history (last 3 rounds) + user
    msgs = [{"role": "system", "content": build_system_prompt(pet)}]
    msgs.extend((history or [])[-6:])
    msgs.append({"role": "user", "content": message})

    body = json.dumps({
        "model":      MODEL,
        "max_tokens": 150,
        "messages":   msgs,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API 错误 {e.code}: {e.read().decode()[:200]}")
    except Exception as e:
        raise RuntimeError(str(e))
