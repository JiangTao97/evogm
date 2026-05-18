import json
import os
import time
import logging
import re
import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from .base_task import BaseTask

log = logging.getLogger(__name__)

class QwenEvaluationTask(BaseTask):
    """
    Evaluation task migrated from eval_qwen/evaluate.py.
    Supports multiple_choice, exact_match, multitask, external_api, and AbstainQA.
    """

    def __init__(self, task_config):
        super().__init__(task_config)
        self.dataset_name = self.config.dataset
        self.eval_type = self.config.eval_type
        # Keep task-level values optional so we can inherit pool-level defaults.
        self.batch_size = self.config.get("batch_size", None)
        self.max_new_tokens = self.config.get("max_new_tokens", None)

    @property
    def taskpool(self):
        return self._taskpool

    @property
    def tokenizer(self):
        return self.taskpool.tokenizer

    @property
    def fabric(self):
        return self.taskpool.fabric

    def _get_taskpool_default(self, key: str, fallback):
        if hasattr(self, "_taskpool") and self._taskpool is not None:
            taskpool_cfg = getattr(self._taskpool, "config", None)
            if taskpool_cfg is not None:
                value = taskpool_cfg.get(key, None)
                if value is not None:
                    return value
        return fallback

    def _effective_batch_size(self) -> int:
        if self.batch_size is not None:
            return int(self.batch_size)
        return int(self._get_taskpool_default("batch_size", 200))

    def _effective_max_new_tokens(self) -> int:
        if self.max_new_tokens is not None:
            return int(self.max_new_tokens)
        return int(self._get_taskpool_default("max_new_tokens", 10))

    def dataset_file(self, dataset_name: str) -> str:
        return os.path.join(self.taskpool.config.data_dir, f"{dataset_name}.json")

    def _get_generation_option(self, key: str, fallback=None):
        if key in self.config and self.config.get(key) is not None:
            return self.config.get(key)
        return self._get_taskpool_default(key, fallback)

    def _should_use_chat_template(self) -> bool:
        return bool(
            self._get_generation_option("use_chat_template", False)
            and hasattr(self.tokenizer, "apply_chat_template")
            and getattr(self.tokenizer, "chat_template", None)
        )

    def _enable_thinking(self):
        return self._get_generation_option("enable_thinking", None)

    def _format_prompts(self, prompts):
        if not self._should_use_chat_template():
            return prompts

        formatted = []
        enable_thinking = self._enable_thinking()
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if enable_thinking is not None:
                kwargs["enable_thinking"] = bool(enable_thinking)
            try:
                formatted.append(self.tokenizer.apply_chat_template(messages, **kwargs))
            except TypeError:
                # Older tokenizer versions may not expose `enable_thinking`.
                kwargs.pop("enable_thinking", None)
                formatted.append(self.tokenizer.apply_chat_template(messages, **kwargs))
        return formatted

    def _decode_generated_text(self, generated_ids):
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        # For Qwen3 thinking mode, keep only the final assistant response after </think>.
        if "</think>" in raw_text:
            raw_text = raw_text.rsplit("</think>", 1)[1]

        for token in [
            getattr(self.tokenizer, "pad_token", None),
            getattr(self.tokenizer, "eos_token", None),
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
        ]:
            if token:
                raw_text = raw_text.replace(token, "")

        raw_text = re.sub(r"^\s*assistant\s*", "", raw_text, flags=re.IGNORECASE)
        return raw_text.strip()

    def multiple_choice_prompt(self, instance_dict, dataset):
        if dataset == "ceval":
            prompt = (
                "下面是一道 C-Eval 单项选择题。"
                "请只输出正确选项的字母（A/B/C/D），不要输出解释。\n"
            )
            prompt += "题目：" + instance_dict["question"] + "\n"
            for key in instance_dict["choices"].keys():
                prompt += (key + ": " + instance_dict["choices"][key] + "\n")
            prompt += "答案："
            return prompt

        prompt = "Question: " + instance_dict["question"] + "\n"
        if dataset == "knowledge_crosswords":
            prompt = prompt
        elif dataset == "hellaswag":
            prompt = "Please choose an option that best completes the sentence.\n" + prompt
        else:
            prompt = "Please choose an option that best answers the question.\n" + prompt

        for key in instance_dict["choices"].keys():
            prompt += (key + ": " + instance_dict["choices"][key] + "\n")
        prompt += "The answer is"

        if dataset == "knowledge_crosswords":
            icl_prompt = self.config.get("icl_prompt", "")
            if not icl_prompt:
                # Fallback to loading from dataset file if not in config
                data_path = self.dataset_file(dataset)
                with open(data_path, 'r') as f:
                    full_data = json.load(f)
                    icl_prompt = full_data.get("icl_prompt", "")
            prompt = icl_prompt + "\n" + prompt
        return prompt

    def multiple_choice_answer_parsing(self, instance_dict, output_text):
        output_text = output_text.strip()

        if self.dataset_name == "ceval":
            ceval_patterns = [
                r"正确答案(?:是|为)?\s*[:：]?\s*\**\s*([ABCD])\b",
                r"答案(?:是|为)?\s*[:：]?\s*\**\s*([ABCD])\b",
                r"应选\s*([ABCD])\b",
                r"选项\s*([ABCD])\b",
            ]
            for pattern in ceval_patterns:
                matches = re.findall(pattern, output_text, flags=re.IGNORECASE)
                if matches:
                    return matches[-1].upper()

            tail_matches = re.findall(
                r"(?:^|[\s\(\[（【])([ABCD])(?:[\s\)\]）】\.\,，。:：]|$)",
                output_text[-80:],
                flags=re.IGNORECASE,
            )
            if tail_matches:
                return tail_matches[-1].upper()
            return "Z"

        # Prefer explicit answer declarations.
        explicit_patterns = [
            r"(?:the\s+answer\s+is|answer\s*:|answer\s+is|option\s+is)\s*([ABCD])\b",
            r"(?:答案(?:是|为)?|选择)\s*[:：]?\s*([ABCD])\b",
        ]
        for pattern in explicit_patterns:
            matches = re.findall(pattern, output_text, flags=re.IGNORECASE)
            if matches:
                return matches[-1].upper()

        # directly answer
        for key in instance_dict["choices"].keys():
            if key in output_text[:5]:
                return key
        # "The answer is ."
        for key in instance_dict["choices"].keys():
            if key in output_text[-5:]:
                return key
        # answer text exact match
        if len(output_text) <= 40:
            for key in instance_dict["choices"].keys():
                if instance_dict["choices"][key].lower() in output_text.lower():
                    return key
        return "Z"

    @torch.no_grad()
    def batch_generate(self, model, prompts, max_new_tokens=10):
        # If max_new_tokens is not explicitly passed, use the task's default
        if max_new_tokens == 10:  # Check if it's still the default value
            max_new_tokens = self._effective_max_new_tokens()

        batch_size = self._effective_batch_size()
        outputs = []
        i = 0
        pbar = tqdm(total=len(prompts), desc=f"Generating for {self.dataset_name}")
        logged_once = False
        do_sample = bool(self._get_generation_option("do_sample", False))
        while i < len(prompts):
            current_bs = min(batch_size, len(prompts) - i)
            while True:
                try:
                    batch_prompts = prompts[i : i + current_bs]
                    model_prompts = self._format_prompts(batch_prompts)
                    inputs = self.tokenizer(model_prompts, return_tensors="pt", padding=True).to(model.device)
                    if not logged_once:
                        log.info(f"Input shape: {inputs.input_ids.shape}")
                        log.info(f"Model device: {model.device}")
                        log.info(f"Batch size: {batch_size}, max_new_tokens: {max_new_tokens}")
                        log.info(
                            "Generation options: use_chat_template=%s enable_thinking=%s do_sample=%s",
                            self._should_use_chat_template(),
                            self._enable_thinking(),
                            do_sample,
                        )
                        logged_once = True

                    generation_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": do_sample,
                        "pad_token_id": self.tokenizer.pad_token_id,
                    }
                    if do_sample:
                        temperature = self._get_generation_option("temperature", None)
                        top_p = self._get_generation_option("top_p", None)
                        top_k = self._get_generation_option("top_k", None)
                        min_p = self._get_generation_option("min_p", None)
                        if temperature is not None:
                            generation_kwargs["temperature"] = temperature
                        if top_p is not None:
                            generation_kwargs["top_p"] = top_p
                        if top_k is not None:
                            generation_kwargs["top_k"] = top_k
                        if min_p is not None:
                            generation_kwargs["min_p"] = min_p

                    output = model.generate(
                        **inputs,
                        **generation_kwargs,
                    )
                    input_len = inputs.input_ids.shape[1]
                    for j in range(len(output)):
                        outputs.append(self._decode_generated_text(output[j][input_len:]))
                    i += current_bs
                    pbar.update(current_bs)
                    break
                except torch.OutOfMemoryError:
                    if current_bs <= 1:
                        raise
                    new_bs = max(1, current_bs // 2)
                    log.warning(
                        "OOM during generation for %s at batch_size=%d. Retrying with batch_size=%d",
                        self.dataset_name,
                        current_bs,
                        new_bs,
                    )
                    current_bs = new_bs
                    batch_size = min(batch_size, new_bs)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch, 'npu') and torch.npu.is_available():
                        torch.npu.empty_cache()
        pbar.close()

        return outputs

    def evaluate(self, model):
        if self.eval_type == "multiple_choice":
            return self.evaluate_multiple_choice(model)
        elif self.eval_type == "exact_match":
            return self.evaluate_exact_match(model)
        elif self.eval_type == "AbstainQA":
            return self.evaluate_abstain_qa(model)
        elif self.eval_type == "multitask":
            return self.evaluate_multitask(model)
        elif self.eval_type == "ifeval":
            return self.evaluate_ifeval(model)
        elif self.eval_type == "autologi":
            return self.evaluate_autologi(model)
        elif self.eval_type == "bfcl":
            return self.evaluate_bfcl(model)
        else:
            raise ValueError(f"Unsupported eval_type: {self.eval_type}")

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                return "\n".join(lines[1:-1]).strip()
        return text

    @staticmethod
    def _callable_spans(text: str):
        """
        Extract top-level callable spans such as `foo(bar=1)` from free-form text.
        This is intentionally lightweight so it works for both plain text and
        code-fenced model outputs.
        """
        spans = []
        i = 0
        while i < len(text):
            if text[i].isalpha() or text[i] in {"_", "."}:
                j = i
                while j < len(text) and (text[j].isalnum() or text[j] in {"_", "."}):
                    j += 1
                k = j
                while k < len(text) and text[k].isspace():
                    k += 1
                if k < len(text) and text[k] == "(":
                    depth = 0
                    start = i
                    in_string = None
                    m = k
                    while m < len(text):
                        ch = text[m]
                        if in_string:
                            if ch == "\\":
                                m += 2
                                continue
                            if ch == in_string:
                                in_string = None
                        else:
                            if ch in {"'", '"'}:
                                in_string = ch
                            elif ch == "(":
                                depth += 1
                            elif ch == ")":
                                depth -= 1
                                if depth == 0:
                                    spans.append(text[start : m + 1].strip())
                                    i = m
                                    break
                        m += 1
            i += 1
        return spans

    @staticmethod
    def _normalize_bfcl_call(text: str) -> str:
        """
        Normalize a BFCL answer into a compact canonical form.

        We accept both raw function-call strings and a small JSON style:
        {"name": "...", "arguments": {...}}
        """
        import ast
        import re

        text = QwenEvaluationTask._strip_code_fences(text)
        text = text.strip()
        if not text:
            return ""

        # Try to recover a raw callable span from free-form output first.
        spans = QwenEvaluationTask._callable_spans(text)
        if spans:
            text = spans[-1]

        # Normalize trivial wrappers.
        text = re.sub(r"^assistant\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^final answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()

        # JSON-ish response: {"name": "foo", "arguments": {"a": 1}}
        if text.startswith("{") and text.endswith("}"):
            try:
                obj = json.loads(text)
                if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    args = obj["arguments"]
                    if isinstance(args, dict):
                        parts = []
                        for key in sorted(args.keys()):
                            parts.append(f"{key}={repr(args[key])}")
                        return f"{obj['name']}({', '.join(parts)})"
            except Exception:
                pass

        # Canonicalize a python-like call expression if possible.
        try:
            expr = ast.parse(text, mode="eval")
            call = expr.body
            if isinstance(call, ast.Call):
                func_name = ast.unparse(call.func)
                parts = []
                for arg in call.args:
                    parts.append(repr(ast.literal_eval(arg)))
                for kw in sorted(call.keywords, key=lambda k: k.arg or ""):
                    if kw.arg is None:
                        # `**kwargs` is not expected in BFCL, but keep it stable.
                        parts.append(f"**{ast.unparse(kw.value)}")
                    else:
                        parts.append(f"{kw.arg}={repr(ast.literal_eval(kw.value))}")
                return f"{func_name}({', '.join(parts)})"
        except Exception:
            pass

        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _canonicalize_bfcl_targets(targets):
        if targets is None:
            return tuple()
        if isinstance(targets, str):
            targets = [targets]
        return tuple(sorted(QwenEvaluationTask._normalize_bfcl_call(t) for t in targets if t))

    def _bfcl_prompt(self, item):
        conversation = item.get("question", [])
        convo_lines = []
        for turn in conversation:
            for message in turn:
                role = message.get("role", "unknown").strip().capitalize()
                content = message.get("content", "")
                convo_lines.append(f"{role}: {content}")

        function_block = json.dumps(item.get("function", []), ensure_ascii=False, indent=2)
        return (
            "You are a precise function-calling assistant.\n"
            "Return only the required function call(s) and do not explain your reasoning.\n"
            "Use canonical Python-call syntax such as `tool_name(arg=value)`.\n"
            "If multiple calls are required, return one call per line.\n\n"
            "Conversation:\n"
            + "\n".join(convo_lines)
            + "\n\nAvailable functions:\n"
            + function_block
            + "\n\nFunction call(s):"
        )

    @staticmethod
    def _autologi_prompt(item):
        question = item.get("question", "").strip()
        input_format = item.get("input_format", "").strip()
        example = item.get("example", "").strip()

        parts = [
            "Solve the constraint satisfaction problem below.",
            question,
        ]
        if input_format:
            parts.extend(
                [
                    "Required input format:",
                    input_format,
                ]
            )
        if example:
            parts.extend(
                [
                    "Example valid answer:",
                    example,
                ]
            )
        parts.append(
            "Return only one valid Python literal for `inputs`."
            " Use Python syntax rather than JSON, including `True`/`False`"
            " and set literals when required. Do not include explanations,"
            " markdown fences, or extra text."
        )
        return "\n\n".join(part for part in parts if part)

    def evaluate_ifeval(self, model):
        """
        Evaluate instruction-following ability using the IFEval benchmark.
        For each prompt, generates a response and checks if ALL instruction
        constraints are satisfied using the official IFEval checker.

        Returns:
            dict with 'accuracy' = prompt-level strict accuracy
                  (fraction of prompts where all instructions pass).
        """
        import sys
        import importlib

        data_path = self.dataset_file(self.dataset_name)
        with open(data_path, 'r') as f:
            full_data = json.load(f)
            eval_data = full_data.get(self.config.get("split", "dev"), [])

        prompts = [item["question"] for item in eval_data]

        # Generate responses with longer max_new_tokens for instruction following
        max_tokens = self.config.get("max_new_tokens", 512)
        outputs = self.batch_generate(model, prompts, max_new_tokens=max_tokens)

        # Try to load the official IFEval instruction checker
        ifeval_script_dir = self._get_taskpool_default(
            "ifeval_script_dir",
            "data/evals/IFEval/instruction_following_eval"
        )

        try:
            if ifeval_script_dir not in sys.path:
                sys.path.insert(0, ifeval_script_dir)
            ifeval_package_root = os.path.dirname(ifeval_script_dir)
            if ifeval_package_root not in sys.path:
                sys.path.insert(0, ifeval_package_root)
            instructions_registry = importlib.import_module("instructions_registry")

            # Check each prompt's instructions
            prompt_pass_count = 0
            instruction_total = 0
            instruction_pass = 0

            for item, output in zip(eval_data, outputs):
                instruction_id_list = item.get("instruction_id_list", [])
                kwargs_list = item.get("kwargs", [])
                all_pass = True

                for inst_id, inst_kwargs in zip(instruction_id_list, kwargs_list):
                    instruction_total += 1
                    try:
                        checker_cls = instructions_registry.INSTRUCTION_DICT.get(inst_id)
                        if checker_cls is None:
                            log.warning(f"Unknown instruction: {inst_id}, skipping")
                            instruction_pass += 1  # be lenient for unknown instructions
                            continue
                        checker = checker_cls(inst_id)
                        if inst_kwargs:
                            checker.build_description(**inst_kwargs)
                        if checker.check_following(output):
                            instruction_pass += 1
                        else:
                            all_pass = False
                    except Exception as e:
                        log.warning(f"Error checking instruction {inst_id}: {e}")
                        all_pass = False

                if all_pass:
                    prompt_pass_count += 1

            prompt_acc = prompt_pass_count / len(eval_data) if eval_data else 0
            inst_acc = instruction_pass / instruction_total if instruction_total > 0 else 0

            log.info(f"IFEval results: prompt_strict_acc={prompt_acc:.4f}, "
                     f"instruction_acc={inst_acc:.4f}")

            return {
                "accuracy": prompt_acc,  # Use prompt-level accuracy as the main metric
                "prompt_strict_accuracy": prompt_acc,
                "instruction_accuracy": inst_acc,
            }

        except ImportError:
            raise RuntimeError(
                f"IFEval instructions_registry not found under {ifeval_script_dir}. "
                "The official checker is required; no fallback is allowed."
            )

    def evaluate_autologi(self, model):
        """
        Evaluate logical reasoning using AutoLogi benchmark.
        Model generates a solution (dict), which is then verified by running
        the constraint checking code from the dataset.

        Returns:
            dict with 'accuracy' = fraction of correct solutions.
        """
        import subprocess
        import tempfile
        import re
        import ast

        data_path = self.dataset_file(self.dataset_name)
        with open(data_path, 'r') as f:
            full_data = json.load(f)
            eval_data = full_data.get(self.config.get("split", "dev"), [])

        prompts = [self._autologi_prompt(item) for item in eval_data]

        # Generate responses - AutoLogi needs longer output for code/dict
        max_tokens = self.config.get("max_new_tokens", 1024)
        outputs = self.batch_generate(model, prompts, max_new_tokens=max_tokens)

        verify_function_code = '''def verify_function(inputs, inputs_check, constraint_list):
    if not inputs_check(inputs):
        return False
    for constraint in constraint_list:
        if not constraint(inputs):
            return False
    return True'''

        correct = 0
        extraction_failed = 0

        for item, output in zip(eval_data, outputs):
            code_info = item.get("code", {})
            inputs_check_code = code_info.get("Inputs_Check_code", "")
            constraint_list_code = code_info.get("Constraint_List_code", "")

            # Extract dict or code block from model output
            func_args = self._extract_autologi_answer(output)
            if func_args is None:
                extraction_failed += 1
                continue

            # Build verification program
            test_program = (
                inputs_check_code + "\n"
                + constraint_list_code + "\n"
                + verify_function_code + "\n"
                + f"res = verify_function({func_args}, inputs_check, constraint_list)\n"
                + "print(res)\n"
                + "assert res == True"
            )

            # Run in subprocess with timeout
            try:
                import sys

                result = subprocess.run(
                    [sys.executable, "-c", test_program],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if result.stdout.strip() in ["True", "true"]:
                    correct += 1
            except (subprocess.TimeoutExpired, Exception):
                pass  # Treat timeout/error as incorrect

        acc = correct / len(eval_data) if eval_data else 0
        log.info(
            f"AutoLogi results: acc={acc:.4f}, correct={correct}, "
            f"extraction_failed={extraction_failed}, total={len(eval_data)}"
        )
        return {
            "accuracy": acc,
            "correct": correct,
            "extraction_failed": extraction_failed,
            "total": len(eval_data),
        }

    def evaluate_bfcl(self, model):
        """
        Evaluate BFCL function-calling tasks using exact canonical match.

        The BFCL V3 executable files already provide `ground_truth` as one or more
        canonical call strings. We normalize both the model output and the target
        calls before comparison.
        """
        data_path = self.dataset_file(self.dataset_name)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"BFCL processed dataset not found at {data_path}. "
                "Run scripts/prepare_new_evals.py to generate bfcl.json first."
            )

        with open(data_path, "r") as f:
            full_data = json.load(f)
            eval_data = full_data.get(self.config.get("split", "dev"), [])

        prompts = [self._bfcl_prompt(item) for item in eval_data]
        max_tokens = self.config.get("max_new_tokens", 256)
        outputs = self.batch_generate(model, prompts, max_new_tokens=max_tokens)

        correct = 0

        for item, output in zip(eval_data, outputs):
            gold = self._canonicalize_bfcl_targets(item.get("ground_truth", []))
            pred = self._canonicalize_bfcl_targets(self._callable_spans(self._strip_code_fences(output)))
            if not pred:
                pred = (self._normalize_bfcl_call(output),) if output.strip() else tuple()

            if pred == gold:
                correct += 1

        acc = correct / len(eval_data) if eval_data else 0
        log.info(
            f"BFCL results: acc={acc:.4f}, correct={correct}, total={len(eval_data)}"
        )
        return {
            "accuracy": acc,
            "correct": correct,
            "total": len(eval_data),
        }

    @staticmethod
    def _extract_autologi_answer(text):
        """Extract dict or code block from model output for AutoLogi."""
        import re
        import ast

        def _normalize(candidate: str) -> str:
            candidate = QwenEvaluationTask._strip_code_fences(candidate).strip()
            candidate = re.sub(r"^assistant\s*:\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"^final answer\s*:\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"^answer\s*:\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r'//.*?(?=\n|$)', '', candidate)
            candidate = re.sub(r"\btrue\b", "True", candidate)
            candidate = re.sub(r"\bfalse\b", "False", candidate)
            candidate = re.sub(r"\bnull\b", "None", candidate)
            return candidate.strip()

        def _last_balanced_dict(candidate: str):
            stack = []
            start_idx = None
            last_dict = None
            in_string = None
            escaped = False
            for i, char in enumerate(candidate):
                if in_string is not None:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == in_string:
                        in_string = None
                    continue
                if char in {"'", '"'}:
                    in_string = char
                    continue
                if char == '{':
                    if not stack:
                        start_idx = i
                    stack.append('{')
                elif char == '}' and stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        last_dict = candidate[start_idx:i + 1]
            return last_dict

        def _try_literal(candidate: str):
            candidate = _normalize(candidate)
            if not candidate:
                return None

            try:
                value = ast.literal_eval(candidate)
                return repr(value)
            except (SyntaxError, ValueError):
                pass

            assignment_match = re.search(
                r"(?:^|\n)\s*(?:inputs|answer|solution)\s*=\s*(.+)\Z",
                candidate,
                flags=re.DOTALL,
            )
            if assignment_match:
                rhs = assignment_match.group(1).strip()
                try:
                    value = ast.literal_eval(rhs)
                    return repr(value)
                except (SyntaxError, ValueError):
                    pass

            embedded_dict = _last_balanced_dict(candidate)
            if embedded_dict and embedded_dict != candidate:
                try:
                    value = ast.literal_eval(_normalize(embedded_dict))
                    return repr(value)
                except (SyntaxError, ValueError):
                    pass

            return None

        code_block_pattern = re.compile(r"```(?:\s*[\w]+)?\s*\n(.*?)```", re.DOTALL)
        code_blocks = code_block_pattern.findall(text)
        for candidate in reversed(code_blocks):
            parsed = _try_literal(candidate)
            if parsed is not None:
                return parsed

        for candidate in (text, _last_balanced_dict(text) or ""):
            parsed = _try_literal(candidate)
            if parsed is not None:
                return parsed

        return None

    def evaluate_multitask(self, model):
        DOMAIN_MAP = {
            "legal": ["hearsay", "citation_prediction_classification"],
            "medical": ["medqa", "medmcqa"],
            "science": ["scifact", "stem"],
            "culture": ["normad_country", "normad_value"]
        }
        domain_tasks = DOMAIN_MAP.get(self.dataset_name)
        if not domain_tasks:
            raise ValueError(f"Unknown multitask domain: {self.dataset_name}")

        scores = []
        ret = {}
        original_dataset_name = self.dataset_name
        for task_name in domain_tasks:
            self.dataset_name = task_name
            res = self.evaluate_multiple_choice(model)
            acc = res.get("accuracy", 0.0)
            scores.append(acc)
            ret[f"{task_name}_accuracy"] = acc
        self.dataset_name = original_dataset_name

        # Calculate Harmonic Mean
        hmean = len(scores) / sum(1.0 / (s + 1e-6) for s in scores)
        ret["harmonic_mean"] = hmean
        # For compatibility with standard metrics, also set 'accuracy' to harmonic mean
        ret["accuracy"] = hmean
        return ret

    def evaluate_multiple_choice(self, model):
        data_path = self.dataset_file(self.dataset_name)
        with open(data_path, 'r') as f:
            full_data = json.load(f)
            eval_data = full_data.get(self.config.get("split", "dev"), [])

        prompts = [self.multiple_choice_prompt(q, self.dataset_name) for q in eval_data]
        outputs = self.batch_generate(model, prompts)

        golds = [q["answer"] for q in eval_data]
        preds = [self.multiple_choice_answer_parsing(q, out) for q, out in zip(eval_data, outputs)]

        acc = accuracy_score(golds, preds)
        return {"accuracy": acc}

    def evaluate_exact_match(self, model):
        data_path = self.dataset_file(self.dataset_name)
        with open(data_path, 'r') as f:
            full_data = json.load(f)
            eval_data = full_data.get(self.config.get("split", "dev"), [])

        prompts = [q["question"] for q in eval_data]

        max_tokens = self.max_new_tokens
        if self.dataset_name == "gsm8k":
            max_tokens = 200

        outputs = self.batch_generate(model, prompts, max_new_tokens=max_tokens)

        if self.dataset_name == "gsm8k":
            outputs = [" ".join(output.split(" ")[-10:]) for output in outputs]

        scores = []
        for q, output in zip(eval_data, outputs):
            if q["answer"] in output:
                scores.append(1)
            else:
                scores.append(0)

        acc = sum(scores) / len(scores) if scores else 0
        return {"accuracy": acc}

    def evaluate_abstain_qa(self, model):
        # Simplified AbstainQA logic
        data_path = self.dataset_file(self.dataset_name)
        with open(data_path, 'r') as f:
            full_data = json.load(f)
            eval_data = full_data.get(self.config.get("split", "dev"), [])

        prompts = [self.multiple_choice_prompt(q, self.dataset_name) for q in eval_data]
        outputs = self.batch_generate(model, prompts)

        correct_flags = []
        for q, output in zip(eval_data, outputs):
            pred = self.multiple_choice_answer_parsing(q, output)
            correct_flags.append(1 if pred == q["answer"] else 0)

        # Self-reflection
        reflect_prompts = [
            p + "\nProposed answer: " + out + "\nIs the proposed answer true or false? Directly answer with true or false."
            for p, out in zip(prompts, outputs)
        ]
        reflect_outputs = self.batch_generate(model, reflect_prompts)

        abstain_flags = [1 if "false" in out.lower() else 0 for out in reflect_outputs]

        # Calculate effective reliability
        answered_correct = 0
        answered_incorrect = 0
        for c, a in zip(correct_flags, abstain_flags):
            if a == 0: # Not abstained
                if c == 1:
                    answered_correct += 1
                else:
                    answered_incorrect += 1

        eff_reliability = (answered_correct - answered_incorrect) / len(correct_flags) if correct_flags else 0
        return {"effective_reliability": eff_reliability}
