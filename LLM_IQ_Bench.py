# LLM 推理选择题基准测试
# 使用方法：加载模型，模型上下文设置在`24576`的同时让显卡最大数量加载模型层数，同时保持空闲显存至少为300MB以上，然后运行程序等待输出结果

import argparse
import difflib
import http.client
import json
import os
import socket
import sys
import time

# 全局变量
DEFAULT_MODEL = "local-model"  # 回退模型名称
HOST = "127.0.0.1"  # 服务端地址
PORT = 1050  # 服务端端口
API_KEY = "llama.cpp-051028..."  # API密钥
TEMPERATURE = 0.95  # 采样温度
OUTPUT_FILENAME = "./Output/LLM_IQ_Bench_Output.json"  # 答题卡

def i(msg): print(f"I| {msg}", flush=True)
def w(msg): print(f"W| {msg}", flush=True)
def e(msg): print(f"E| {msg}", flush=True)

def ask_yes_no(prompt):
    """交互式 yes/no 询问；非 TTY 环境默认继续"""
    try:
        if not sys.stdin.isatty():
            i(f"{prompt} -> 非交互环境，默认继续")
            return True
    except Exception:
        return True
    while True:
        try:
            ans = input(f"{prompt} (yes/no): ").strip().lower()
        except EOFError:
            return True
        if ans in ("yes", "y"):
            return True
        if ans in ("no", "n"):
            return False
        print("请输入 yes 或 no")

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

class CotLoopDetector:
    """检测 CoT 尾部是否出现连续 repeat 段模糊重复"""

    def __init__(self, threshold=0.95, repeat=3,
                 chunk_sizes=(150, 250, 400, 600),
                 check_interval=40, window=20000):
        self.threshold = threshold
        self.repeat = repeat
        self.chunk_sizes = tuple(chunk_sizes)
        self.check_interval = check_interval
        self.window = window
        self.buf = ""
        self._since_last_check = 0

    def feed(self, text):
        if not text:
            return False
        self.buf += text
        if len(self.buf) > self.window:
            self.buf = self.buf[-self.window:]
        self._since_last_check += len(text)
        if self._since_last_check < self.check_interval:
            return False
        self._since_last_check = 0
        return self.check()

    def _sim(self, a, b):
        sm = difflib.SequenceMatcher(None, a, b)
        if sm.quick_ratio() < self.threshold:
            return 0.0
        return sm.ratio()

    def check(self):
        n = len(self.buf)
        r = self.repeat
        for cs in self.chunk_sizes:
            need = cs * r
            if n < need:
                continue
            segs = [self.buf[n - need + k * cs: n - need + (k + 1) * cs]
                    for k in range(r)]
            if all(self._sim(segs[k], segs[k + 1]) >= self.threshold
                   for k in range(len(segs) - 1)):
                return True
        return False

def load_questions(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        e(f"题库文件不存在: {path}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        e(f"题库 JSON 解析失败: {exc}")
        sys.exit(1)

    if not isinstance(data, list) or not data:
        e("题库根节点必须是非空数组")
        sys.exit(1)

    for idx, q in enumerate(data):
        for key in ("id", "question", "options", "answer"):
            if key not in q:
                e(f"第 {idx + 1} 条缺少字段: {key}")
                sys.exit(1)
        if len(q["options"]) != 4:
            e(f"第 {idx + 1} 条选项数不是 4")
            sys.exit(1)
        ans = str(q["answer"]).upper()
        if ans not in ("A", "B", "C", "D"):
            e(f"第 {idx + 1} 条答案非法: {q['answer']}")
            sys.exit(1)
        q["answer"] = ans
    return data

def build_prompt(q):
    lettered = "\n".join(f"{chr(65 + idx)}. {opt}" for idx, opt in enumerate(q["options"]))
    return (
        f"{q['question']}\n\n{lettered}\n\n"
        "请只回答正确选项的单个字母，不要输出任何其他内容"
    )

def build_request_body(model, q, temperature):
    return {
        "model": model,
        "messages": [
            # 系统提示：约束模型只回答单个字母
            {"role": "system",
             "content": "你是一个友好的 AI 助手，回答问题时只需要回答单个字母，不要做任何解释"},
            {"role": "user", "content": build_prompt(q)},
        ],
        "temperature": temperature,
        # 与模型上下文设置一致的最大生成 token 数
        "max_tokens": 24576,
        "stream": True,  # 流式输出，防止生成时静默导致超时
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

def ask_one(host, port, api_key, model, timeout, question, temperature):
    body_bytes = json.dumps(
        build_request_body(model, question, temperature), ensure_ascii=False
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream",
        "Connection": "keep-alive",
    }

    result = {
        "content": [], "reasoning": [],
        "error": None, "loop_detected": False,
    }
    loop_detector = CotLoopDetector()

    conn = None
    t_start = time.perf_counter()
    t_done = None
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.connect()
        try:
            conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

        conn.request("POST", "/v1/chat/completions", body=body_bytes, headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            err = resp.read().decode("utf-8", errors="replace")
            result["error"] = f"HTTP {resp.status}: {err}"
        else:
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
                    break
                if not obj.get("choices"):
                    continue
                delta = obj["choices"][0].get("delta") or {}

                rc = delta.get("reasoning_content") or delta.get("reasoning")
                if rc:
                    result["reasoning"].append(rc)
                    if loop_detector.feed(rc):
                        # 强制打断当前请求，跳出 SSE 读取循环
                        result["loop_detected"] = True
                        break

                c = delta.get("content")
                if c:
                    result["content"].append(c)

    except socket.timeout:
        result["error"] = f"超时({timeout}s)"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    if t_done is None:
        t_done = time.perf_counter()
    result["elapsed"] = t_done - t_start
    return result

def parse_answer(text):
    """返回 A/B/C/D；解析失败返回 None"""
    if not text:
        return None
    s = text.strip().upper()
    s = s.strip("*").strip("()[]【】")
    s = s.rstrip(".,;:!?。，；：！？…").strip()
    return s if s in ("A", "B", "C", "D") else None

# 断点续传相关
def atomic_write_json(path, obj):
    """先写临时文件再原子替换，避免中途崩溃留下半截文件"""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, path)
        return True
    except Exception as exc:
        w(f"写入 {path} 失败: {exc}")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False

def backup_file(path, suffix="bak"):
    if not os.path.exists(path):
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = f"{path}.{ts}.{suffix}"
    try:
        os.replace(path, dst)
        i(f"旧文件已备份为 {dst}")
    except Exception as exc:
        w(f"备份 {path} 失败: {exc}")

def load_progress(path, model):
    """返回进度字典；不存在 / 损坏 / 模型不匹配时返回 None"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        w(f"进度文件解析失败: {exc}")
        backup_file(path, "corrupt")
        return None
    if not isinstance(data, dict):
        w("进度文件结构不合法，忽略")
        backup_file(path, "invalid")
        return None
    prev_model = data.get("model")
    if prev_model != model:
        w(f"进度文件模型不匹配(保存={prev_model}，当前={model})，从头开始")
        backup_file(path, "model_mismatch")
        return None
    return data

def save_progress(path, model, questions_path, total, records, counters):
    payload = {
        "version": 1,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "questions": os.path.abspath(questions_path),
        "total": total,
        "done": len(records),
        "counters": counters,
        "records": records,
    }
    atomic_write_json(path, payload)

TAG = {
    "correct": "✓ 正确",
    "wrong": "✗ 答错",
    "parse_failed": "✗ 解析失败",
    "request_error": "✗ 请求错误",
    "loop_failed": "✗ CoT循环失败",
}

def main():
    global HOST, PORT, API_KEY

    ap = argparse.ArgumentParser(description="LLM 推理选择题基准测试")
    ap.add_argument("--host", default=HOST, help="服务端地址(默认取全局 HOST)")
    ap.add_argument("--port", type=int, default=PORT, help="服务端端口(默认取全局 PORT)")
    ap.add_argument("--api-key", default=API_KEY, help="API Key(默认取全局 API_KEY)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--questions", default="Questions.json", help="题库 JSON 路径")
    ap.add_argument("--save", default="./Output/LLM_IQ_Bench_Grades.txt",
                    help="成绩单保存路径")
    ap.add_argument("--timeout", type=float, default=1200, help="单次请求超时(秒)")
    ap.add_argument("--loop-max-attempts", type=int, default=3,
                    help="同一题累计 CoT 循环判错的最大尝试次数(默认 3)")
    ap.add_argument("--parse-max-attempts", type=int, default=2,
                    help="解析失败时最多发送题目的次数(含首次，默认 2)")
    ap.add_argument("--progress", default="LLM_IQ_Bench_Progress.json",
                    help="断点续传进度文件路径(默认当前目录)")
    ap.add_argument("--retry-failed-on-resume", action="store_true",
                    help="续传时重新作答上次非 correct 状态的题目")
    args = ap.parse_args()

    # 将命令行实际取值同步回全局变量，保证全局与运行时一致
    HOST = args.host
    PORT = args.port
    API_KEY = args.api_key

    # 过程输出与成绩单所在目录
    os.makedirs("./Output", exist_ok=True)

    questions = load_questions(args.questions)
    total = len(questions)
    per_q = 100.0 / total

    model = args.model
    if model == DEFAULT_MODEL:
        auto = fetch_model_name(HOST, PORT, API_KEY, model)
        if auto != model:
            model = auto
            i(f"自动获取模型名: {model}")

    i("=" * 62)
    i(f"目标: http://{HOST}:{PORT}/v1/chat/completions")
    i(f"模型: {model}")
    i(f"题库: {args.questions}  ({total} 题)")
    i(f"过程输出: {OUTPUT_FILENAME}")
    i(f"成绩单: {args.save}")
    i(f"进度文件: {args.progress}")
    i(f"CoT循环判错阈值: 连续 {args.loop_max_attempts} 次")
    i(f"解析失败重试上限: 共 {args.parse_max_attempts} 次尝试")
    i(f"每题权重: {per_q:.2f}  满分: 100.00")
    i("=" * 62)

    # 加载断点续传进度：检测到文件时交互式询问
    progress = None
    if os.path.exists(args.progress):
        if ask_yes_no(f"检测到进度文件 {args.progress}，是否继续上次的进度？"):
            progress = load_progress(args.progress, model)
        else:
            backup_file(args.progress, "noresume")
            i("已忽略旧进度，从头开始")
    else:
        i("未发现进度文件，从头开始")

    done_map = {}
    if progress:
        done_records = progress.get("records") or []
        prev_total = progress.get("total")
        if prev_total and prev_total != total:
            w(f"题库数量变化：上次 {prev_total} 题，本次 {total} 题，按题目 id 续传")
        for rec in done_records:
            if isinstance(rec, dict) and rec.get("id") is not None:
                done_map[rec["id"]] = rec
        i(f"发现匹配进度，已有 {len(done_map)} 条记录，将从中断处继续")

    records = []
    counters = {"correct": 0, "wrong": 0, "parse_failed": 0,
                "request_error": 0, "loop_failed": 0}

    try:
        for idx, q in enumerate(questions, 1):
            qid = q["id"]

            if qid in done_map:
                prev = done_map[qid]
                prev_status = prev.get("status")
                if not (args.retry_failed_on_resume and prev_status != "correct"):
                    # 补全旧进度记录中可能缺失的字段
                    prev.setdefault("question", q["question"])
                    prev.setdefault("options", q["options"])
                    records.append(prev)
                    if prev_status in counters:
                        counters[prev_status] += 1
                    print(f"[{idx}/{total}] 已完成，跳过  |  答对 {counters['correct']}/{idx}  "
                          f"|  {TAG.get(prev_status, str(prev_status))}", flush=True)
                    continue
                i(f"第 {qid} 题上次状态={prev_status}，按 --retry-failed-on-resume 重做")

            print(f"[{idx}/{total}] 处理中...  已答对 {counters['correct']}", flush=True)

            loop_attempts = 0    # 累计检测到 CoT 循环的次数
            parse_attempts = 0   # 解析失败累计次数
            attempt_elapsed = 0.0
            got = None

            while True:
                # 固定温度，不提升
                r = ask_one(HOST, PORT, API_KEY, model,
                            args.timeout, q, TEMPERATURE)
                attempt_elapsed += r["elapsed"]

                if r["error"]:
                    # 请求错误：直接按原流程处理
                    break

                if r["loop_detected"]:
                    loop_attempts += 1
                    if loop_attempts >= args.loop_max_attempts:
                        w(f"第 {q['id']} 题连续 {loop_attempts} 次 CoT 循环，"
                          f"判定失败并进入下一题")
                        break
                    w(f"第 {q['id']} 题 CoT 检测到循环（第 {loop_attempts} 次），"
                      f"已打断当前请求，以相同温度 {TEMPERATURE:.2f} 重新提问")
                    continue

                # 无循环，尝试解析
                got = parse_answer("".join(r["content"]))
                if got is None:
                    parse_attempts += 1
                    if parse_attempts >= args.parse_max_attempts:
                        w(f"第 {q['id']} 题第 {parse_attempts} 次解析失败，"
                          f"按原流程记录为 parse_failed")
                        break
                    w(f"第 {q['id']} 题第 {parse_attempts} 次解析失败，"
                      f"重新发送同一道题目")
                    continue

                # 解析成功
                break

            content = "".join(r["content"])
            reasoning = "".join(r["reasoning"])
            elapsed = round(attempt_elapsed, 2)

            if r["error"]:
                status = "request_error"; got = None
            elif r["loop_detected"]:
                status = "loop_failed"; got = None
            elif got is None:
                status = "parse_failed"
            elif got == q["answer"]:
                status = "correct"
            else:
                status = "wrong"

            # 最终若仍处于循环判错，最后一次尝试不算“重试”
            loop_retries = loop_attempts - (1 if r["loop_detected"] else 0)

            record = {
                "id": q["id"], "question": q["question"], "options": q["options"],
                "expected": q["answer"], "got": got, "status": status,
                "parse_failed": status not in ("correct", "wrong"),
                "loop_retries": loop_retries,
                "loop_detected_final": r["loop_detected"],
                "elapsed": elapsed, "content": content, "reasoning": reasoning,
            }
            records.append(record)
            counters[status] += 1

            print(f"[{idx}/{total}] 完成  |  答对 {counters['correct']}/{idx}  |  {TAG[status]}  "
                  f"|  循环重试 {loop_retries} 次  |  {elapsed}s", flush=True)

            atomic_write_json(OUTPUT_FILENAME, records)
            save_progress(args.progress, model, args.questions, total, records, counters)

    except KeyboardInterrupt:
        w("收到中断信号，正在保存进度...")
        try:
            save_progress(args.progress, model, args.questions, total, records, counters)
            i(f"进度已保存到 {args.progress}，下次运行将继续")
        except Exception as exc:
            w(f"保存进度失败: {exc}")
        return 130

    correct = counters["correct"]
    wrong = counters["wrong"]
    parse_failed = counters["parse_failed"]
    request_error = counters["request_error"]
    loop_failed = counters["loop_failed"]

    score = round(correct * per_q, 2)

    print("\n" + "=" * 62)
    i("最终结果")
    print("=" * 62)
    for rec in records:
        i(f"  第 {rec['id']:>2} 题  {TAG[rec['status']]:<12}  "
          f"期望={rec['expected']}  实际={rec['got'] or 'N/A'}  "
          f"循环重试={rec['loop_retries']}  耗时={rec['elapsed']}s")
    print("-" * 62)
    i(f"  答对:     {correct}/{total}")
    i(f"  答错:     {wrong}/{total}")
    i(f"  解析失败: {parse_failed}/{total}")
    i(f"  请求错误: {request_error}/{total}")
    i(f"  循环失败: {loop_failed}/{total}")
    print("-" * 62)
    i(f"  总分: {score:.2f}/100")
    print("=" * 62)

    now_str = time.strftime("%Y年%m月%d日%H时%M分", time.localtime())
    card_lines = []
    for n, rec in enumerate(records, 1):
        mark = "✓" if rec["status"] == "correct" else "✗"
        got = rec["got"] or "-"
        card_lines.append(
            f"{n}. 作答: {got}  正确: {rec['expected']}  {mark}  [{rec['status']}]"
        )

    try:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(f"结束时间: {now_str}\n")
            f.write(f"分数: {score:.2f}/100\n")
            f.write(f"答对: {correct}/{total}\n")
            f.write("答题卡:\n")
            f.write("\n".join(card_lines))
            f.write("\n")
        i(f"成绩单已保存到 {args.save}")
    except Exception as exc:
        w(f"保存失败: {exc}")

    # 全部完成，可选删除进度文件
    i(f"进度文件保留在 {args.progress}(如需从头开始删除它)")
    return 0

if __name__ == "__main__":
    sys.exit(main())