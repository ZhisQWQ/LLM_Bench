# LLM 输出速度测试
# 使用方法：加载模型，模型上下文设置在`4096`的同时让显卡最大数量加载模型层数，同时保持空闲显存至少为300MB以上，然后运行程序等待输出结果

import argparse
import difflib
import http.client
import json
import os
import socket
import sys
import time

# 全局变量
HOST = "127.0.0.1"  # 服务端地址
PORT = 1050  # 服务端端口
API_KEY = "llama.cpp-051028..."  # API密钥
MODEL_NAME = "local-model"  # 回退模型名称
TEMPERATURE = 0.95  # 采样温度
OUTPUT_DIR = "./Output"  # 测试结果
DEFAULT_SAVE = os.path.join(OUTPUT_DIR, "LLM_Speed_Output.txt")  # 测试结果

# 系统提示词
SYSTEM_PROMPT = "你是一个友好的 AI 助手，跟随用户语言回复用户"

PROMPT = """
请完整的复述以下内容，不需要做任何更改也不需要任何提示：
<txt>
常常一个人静静的呆在一个角落，不想被各种繁杂所惊扰，给心足够的呼吸。宁静，凝固了岁月，静止了时间，品味孤独带来的淡淡忧伤。孤独是灵与肉的交汇，孤独是净化心灵的唯美。孤独让我们静下心来思考人情的冷暖、生命的真诚。孤独让我们理清了杂乱的脉络、悟透了生活的精髓。孤独是一脉清泉的纯净，是一粒种子的重生，孤独让心灵沉稳，让生命坚定，孤独强化了生存的本能。
总有一些人离开了视线，却留下一段往事，总有一些背影已经走远，却住进了生命。尘缘若梦起伏不平，总在用一些宽慰的词汇来为自己解脱，用虚伪的面具来包装成对尘世的淡泊，其实我们都很脆弱，谁又明了伪装后的坚强包含的苦涩，不经意的触摸就会洞穿心底。总是制造喧嚣的气氛来掩盖内心的孤独，曲散楼空却依旧是一个人的落寞，秋水不懂落叶的凄婉，夜风又怎知云的哀叹，只有无眠的雨滴敲打着岁月的阑珊，听时光在诉说。
落落红尘如烟往事，几经辗转沉浮，数尽尘世悲欢，感叹红尘情深缘浅；他日双影映斜阳，星月伴呢喃，也许今朝执笔写寒凉，落寞独倚窗。是岁月的脚步走得太快，还是转身太过匆忙？一念便是地角天涯，回眸以是山高水长。尘烟渐退，终究握不住倾城繁华，时光划过，徒留馨香几许？一道岁月的栏栅屏障了相同的季节，盈香满袖，却触摸不到花期，一叶扁舟难以承载满腹心事，此岸彼岸，隔着无法摆渡的距离，只有暗香一缕若有若无的飘散在风尘里。经年走过，轻轻念起依然会有些许的疼痛。
有时候莫名的不知所措，只想觅一方寸之地，静静的感受心跳和呼吸，把不安的躁动和低落的情绪压在心底，轻轻的安抚，远离世俗的惊扰，安逸一份平淡的心性。没有心事的神伤，常常会制造一种不知所以的挤压与彷徨，困扰心情的也许不是得失，而是一种不知名的感伤，一种灵魂的空旷。而我们常在这种感觉中无法驾驭自我，不经意间就触摸到了一种不知名的脆弱。也许只有安静才能给予心灵极致的净化，才能走出自己设下的漩涡。
我们在一段时光里笑过、哭过，一时排解不开纠结的情绪，随着时光的流逝慢慢沉淀在心底，汇集成了一道雕琢的风景。岁月掠夺了我们的容颜，却留下了属于我们的故事。有些故事虽然只是刹那芳华，却趟过了岁月的心河，融入了血液。在与时光的对望中，嫣然了笑容，泪流了感动，经年走过，我们驻足、回望，却依然心存流连、无怨无悔。
总有一段时光孤单了一段路程，或许有很多话想说，却又不清楚该说什么，该向谁说。也许会莫名的失落，做一些违反常态的举动证明自己存在着。也许喜欢上了孤独，习惯了默默的去承受一切。其实人生并不孤单，只是被某一处阴影遮住了阳光的灼热，而忽略了关心我们的每一双眼睛。四季流转，不同的风景给生命做着无私的衬托，原来每一份关心都来自情怀默默。旅途无期，一声珍重，千言万语的感动。一簇小草，一株野花，一片枯叶，无不彰显生命的厚重。人生本不悲凉，拉开紧固的衣襟，敞开胸怀去接纳阳光，拥抱每一处暖心的风景，热度就会贯穿整个血脉，生命就会充满温情。
一个人静静的听一首歌，品一杯茶，在空灵幽雅的旋律中舒展所有的心事，抛开人情的冷凉，以一曲歌韵释怀所有的心伤。当时光掠过发髻，轻舞衣衫，抖落满身微尘雾雨，所有的快乐和忧伤都无我无关。让心如止水，以一朵野菊花的姿态默守一份淡淡然。任流年荡过指尖，不提及往事，不言论悲欢。
岁月就像跳动的音符，穿过季节的斑驳，奏响着时光的章程，谱曲着一首首生命之歌。在一份寂静中沉淀自己的心性，细细倾听一首轻柔的音乐，任旋律穿透肺腑融入血脉。什么都不去想，一切烦乱被剥落，岁月仿佛静止，时光在这一刻凝固。心随旋律飘飞，荡游在太空，穿越在无我的幻境中，无争无扰、无欲无求。闭上眼睛，感受心跳和呼吸，醉了。胸怀瞬息被放大，包容了自己，包容了世界，释怀了繁杂。一条河流过，流进生命，穿过了风尘，趟过了岁月，轻轻地、轻轻地流淌着。
拨开渐退的雾霾，面对晨起的朝阳，舒展怀抱盈一缕温暖。踏入一席清新之地，赏心满目青绿，取一滴晨露品其甘甜，野花轻吐嫣蕊低眉浅笑，池柳轻摇衣带飘飘。绿草随风踏浪，风过风停静默一份淡泊。小桥流水清澈依然，环山而绕清逸而悠远，看流云隐逸远山的羞涩，倾听鸟儿嬉戏云水间。任时光走过指尖，将晴好尽数打点，拾一份温润入心，守一份安然随情。
踏过冰雪的轮回，脱下旧日薄凉，嗅着春天的气息，播下一粒快乐的种子，用温润孕育让它结满温情，风起风停笑对人生，月缺月圆淡看旅程，孤单的日子，拾起一缕明媚，与温暖同行。旅途匆匆紧握一份感动，沧海桑田珍重一路随行，生命的价值是一种紧握，风雨路上依旧芳菲，人生的完美是携手冷暖，同看落日余晖，与时光安然相处，默守岁月赋予的唯美.
</txt>
"""

def i(msg): print(f"I| {msg}", flush=True)
def w(msg): print(f"W| {msg}", flush=True)
def e(msg): print(f"E| {msg}", flush=True)

def fetch_model_name(host, port, api_key, fallback):
    """通过 /v1/models 自动获取第一个模型名，失败时回退"""
    conn = None
    try:
        conn = http.client.HTTPConnection(host, port, timeout=10)
        conn.request("GET", "/v1/models",
                     headers={"Authorization": f"Bearer {api_key}"})
        resp = conn.getresponse()
        if resp.status == 200:
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("data") or []
            if items and items[0].get("id"):
                return items[0]["id"]
    except Exception:
        pass
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return fallback

def build_request_body(model):
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": 4096,           # 硬编码
        "stream": True,               # 流式输出，防止生成时静默导致超时
        "stream_options": {"include_usage": True},
    }

def parse_sse_line(line):
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        return "__DONE__"
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None

def extract_reference():
    """从 PROMPT 中抽取 <txt>...</txt> 作为参考原文"""
    start = PROMPT.find("<txt>")
    end = PROMPT.find("</txt>")
    if start == -1 or end == -1 or end <= start:
        return None
    return PROMPT[start + len("<txt>"):end]

def _normalize(s):
    return "".join(s.split())

def similarity(a, b):
    a = _normalize(a)
    b = _normalize(b)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

def fmt(dt):
    return "N/A" if dt is None else f"{dt:.3f}s"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=DEFAULT_SAVE, help="保存模型输出路径")
    ap.add_argument("--no-save", action="store_true", help="不保存输出")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path = "/v1/chat/completions"
    model = fetch_model_name(HOST, PORT, API_KEY, MODEL_NAME)
    if model != MODEL_NAME:
        i(f"自动获取模型名: {model}")

    body_bytes = json.dumps(build_request_body(model), ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "text/event-stream",
        "Connection": "keep-alive",
    }

    i(f"目标: http://{HOST}:{PORT}{path}")
    i(f"模型: {model}  temperature={TEMPERATURE}  max_tokens=4096")
    i("发送请求 ...")

    t_start = t_headers = t_first_token = None
    t_first_reasoning = t_first_content = None
    t_done = None

    reasoning_chunks = content_chunks = 0
    reasoning_text = []
    content_text = []
    usage = None
    got_done = False

    conn = None
    try:
        conn = http.client.HTTPConnection(HOST, PORT, timeout=1200)  # 硬编码
        conn.connect()
        try:
            conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

        t_start = time.perf_counter()
        conn.request("POST", path, body=body_bytes, headers=headers)
        resp = conn.getresponse()
        t_headers = time.perf_counter()
        i(f"响应头到达 status={resp.status} t={fmt(t_headers - t_start)}")

        if resp.status != 200:
            e(f"HTTP {resp.status}: {resp.read().decode('utf-8', errors='replace')}")
            return 1

        while True:
            raw = resp.readline()
            if not raw:
                break
            now = time.perf_counter()

            obj = parse_sse_line(raw.decode("utf-8", errors="replace"))
            if obj is None:
                continue

            if obj == "__DONE__":
                t_done = now
                got_done = True
                i(f"完成 [DONE] t={fmt(t_done - t_start)}")
                break

            if obj.get("usage"):
                usage = obj["usage"]
                i(f"usage t={fmt(now - t_start)} {json.dumps(usage, ensure_ascii=False)}")

            if not obj.get("choices"):
                continue
            delta = obj["choices"][0].get("delta") or {}

            rc = delta.get("reasoning_content") or delta.get("reasoning")
            if rc:
                reasoning_chunks += 1
                reasoning_text.append(rc)
                if t_first_reasoning is None:
                    t_first_reasoning = now
                    i(f"首个 reasoning token t={fmt(now - t_start)}")
                if t_first_token is None:
                    t_first_token = now
                    i(f"首个 token t={fmt(now - t_start)}")

            c = delta.get("content")
            if c:
                content_chunks += 1
                content_text.append(c)
                if t_first_content is None:
                    t_first_content = now
                    gap = (now - t_first_token) if t_first_token else 0.0
                    i(f"首个 content token t={fmt(now - t_start)} "
                      f"(距首 token {fmt(gap)})")
                if t_first_token is None:
                    t_first_token = now
                    i(f"首个 token t={fmt(now - t_start)}")

    except socket.timeout:
        e("超时(1200s)")
        return 1
    except Exception as exc:
        e(f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    if t_first_token is None:
        w("没有收到任何 token")
        return 1
    if t_done is None:
        t_done = time.perf_counter()

    if usage and isinstance(usage.get("completion_tokens"), (int, float)):
        total_tokens = int(usage["completion_tokens"])
        token_src = "usage.completion_tokens(服务端精确，含 CoT)"
    elif usage and isinstance(usage.get("total_tokens"), (int, float)):
        total_tokens = int(usage["total_tokens"])
        token_src = "usage.total_tokens(服务端精确)"
    else:
        total_tokens = reasoning_chunks + content_chunks
        token_src = "chunk 数兜底(非精确)"

    ttft = t_first_token - t_start
    gen_time = max(t_done - t_first_token, 1e-9)
    total_time = t_done - t_start
    speed = total_tokens / gen_time
    e2e = total_tokens / total_time

    cot_time = ans_time = None
    if t_first_reasoning and t_first_content:
        cot_time = t_first_content - t_first_reasoning
        ans_time = t_done - t_first_content
    elif t_first_reasoning:
        cot_time = t_done - t_first_reasoning
    elif t_first_content:
        ans_time = t_done - t_first_content

    i("=" * 62)
    i("核心结果")
    i("=" * 62)
    i(f"  输出 Token 总数        : {total_tokens}  ({token_src})")
    i(f"    CoT chunk 数         : {reasoning_chunks}")
    i(f"    正文 chunk 数        : {content_chunks}")
    i(f"  首 Token 延迟 (TTFT)   : {fmt(ttft)}")
    i(f"  Token 流持续时长       : {fmt(gen_time)}   (首个 token → 结束)")
    i(f"  端到端总耗时           : {fmt(total_time)}   (请求 → 结束)")
    i("-" * 62)
    i(f"  生成速度 (CoT + 正文)  : {speed:.2f} tokens/s")
    i(f"      = {total_tokens} tokens / {gen_time:.3f}s")
    i("-" * 62)
    i(f"  参考：端到端吞吐       : {e2e:.2f} tokens/s")
    if cot_time is not None or ans_time is not None:
        i("-" * 62)
        if cot_time is not None:
            i(f"  CoT 阶段   : {fmt(cot_time)}  chunks={reasoning_chunks}")
        if ans_time is not None:
            i(f"  正文阶段   : {fmt(ans_time)}  chunks={content_chunks}")
    i("=" * 62)

    passed = None
    ref = extract_reference()
    if ref is None:
        w("未找到 <txt>...</txt> 参考内容，跳过相似度比对")
    else:
        ratio = similarity("".join(content_text), ref)
        i(f"相似度比对: {ratio * 100:.2f}%  (阈值 98%)")
        if ratio >= 0.98:
            i("比对结果: 通过")
            passed = True
        else:
            w(f"比对结果: 不通过({ratio * 100:.2f}% < 98%)")
            passed = False

    if not got_done:
        w("未收到 [DONE] 标记，流可能被截断")

    if not args.no_save and (reasoning_text or content_text):
        try:
            with open(args.save, "w", encoding="utf-8") as f:
                if reasoning_text:
                    f.write("=" * 30 + " CoT " + "=" * 30 + "\n")
                    f.write("".join(reasoning_text))
                    f.write("\n\n" + "=" * 28 + " 正文 " + "=" * 28 + "\n")
                f.write("".join(content_text))
            i(f"输出已保存到 {args.save}")
        except Exception as exc:
            w(f"保存失败: {exc}")

    return 2 if passed is False else 0

if __name__ == "__main__":
    sys.exit(main())