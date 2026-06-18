"""One-command fine-tuning: /ft <dataset> [base_model]"""
import os, sys, json, subprocess
from pathlib import Path

BASE = Path(os.path.expanduser("~/rsa-agentic"))
FT_DIR = BASE / "fine_tuned"
FT_DIR.mkdir(parents=True, exist_ok=True)

def detect_gpu():
    result = {"available": False, "type": "cpu", "name": "CPU"}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            gpu_name = r.stdout.strip().split("\n")[0].split(",")[0]
            result = {"available": True, "type": "cuda", "name": gpu_name}
    except:
        pass
    if not result["available"]:
        try:
            import torch
            if torch.backends.mps.is_available():
                result = {"available": True, "type": "mps", "name": "Apple Silicon"}
        except:
            pass
    return result

def run(ds_path, base_model="qwen3:0.6b", output_name=""):
    gpu = detect_gpu()
    oname = output_name or f"ft-{Path(ds_path).stem}"
    script_path = FT_DIR / f"train_{oname}.py"
    
    # Generate script content
    content = [
        "#!/usr/bin/env python3",
        "# RSA Agentic Fine-Tuning Script",
        'import json, os, sys, subprocess',
        'from pathlib import Path',
        '',
        f'MODEL = sys.argv[1] if len(sys.argv) > 1 else "{base_model}"',
        f'DATA = sys.argv[2] if len(sys.argv) > 2 else "{ds_path}"',
        f'OUT = sys.argv[3] if len(sys.argv) > 3 else "{FT_DIR / oname}"',
        f'NAME = "{oname}"',
        '',
        'try:',
        '    from unsloth import FastLanguageModel',
        '    import torch',
        '    from datasets import Dataset',
        '    from trl import SFTTrainer',
        '    from transformers import TrainingArguments',
        '',
        '    samples = []',
        '    with open(DATA) as f:',
        '        if DATA.endswith(".jsonl"):',
        '            for line in f:',
        '                if line.strip(): samples.append(json.loads(line))',
        '        else: samples = json.load(f)',
        '',
        '    texts = []',
        '    for s in samples[:100]:',
        '        msgs = s.get("messages", [])',
        '        if msgs:',
        '            text = ""',
        '            for m in msgs:',
        '                role = m.get("role", "")',
        '                content = m.get("content", "")[:500]',
        '                if role == "user": text += f"### Human: {content}\\n"',
        '                elif role == "assistant": text += f"### Assistant: {content}\\n"',
        '            text += "### Assistant:"',
        '            texts.append({"text": text})',
        '',
        '    if not texts:',
        '        print("No valid conversations")',
        '        sys.exit(1)',
        '',
        '    print(f"Loaded {len(texts)} samples")',
        '    model, tokenizer = FastLanguageModel.from_pretrained(',
        '        MODEL, max_seq_length=2048, load_in_4bit=True',
        '    )',
        '    model = FastLanguageModel.get_peft_model(',
        '        model, r=16, lora_alpha=16,',
        '        target_modules=["q_proj","k_proj","v_proj","o_proj"],',
        '        lora_dropout=0, bias="none", use_gradient_checkpointing=True',
        '    )',
        '    trainer = SFTTrainer(',
        '        model=model, tokenizer=tokenizer,',
        '        train_dataset=Dataset.from_list(texts),',
        '        args=TrainingArguments(',
        '            output_dir=OUT, per_device_train_batch_size=2,',
        '            num_train_epochs=1, logging_steps=10,',
        '            learning_rate=2e-4,',
        '            fp16=torch.cuda.is_available(),',
        '            report_to="none",',
        '        ),',
        '    )',
        '    trainer.train()',
        '    model.save_pretrained(OUT)',
        '    tokenizer.save_pretrained(OUT)',
        '    Path(f"{OUT}/Modelfile").write_text(f"FROM {OUT}\\n")',
        '    r = subprocess.run(["ollama","create",NAME,"-f",f"{OUT}/Modelfile"],',
        '        capture_output=True, text=True, timeout=300)',
        '    if r.returncode == 0:',
        '        print(f"Uploaded to Ollama: {NAME}")',
        '        # Auto-switch model',
        '        import tomli_w',
        '        cfg = tomllib.load(open(os.path.expanduser("~/rsa-agentic/config.toml"),"rb"))',
        '        cfg["model"]["model_name"] = NAME',
        '        with open(os.path.expanduser("~/rsa-agentic/config.toml"),"wb") as f:',
        '            tomli_w.dump(cfg, f)',
        '        print(f"Switched to: {NAME}")',
        '    else:',
        '        print(f"Upload: {r.stderr[:200]}")',
        'except ImportError as e:',
        '    print(f"Install: pip install unsloth transformers datasets trl")',
        'except Exception as e:',
        '    print(f"Error: {e}")',
        '    import traceback',
        '    traceback.print_exc()',
    ]
    
    script_path.write_text("\n".join(content))
    os.chmod(str(script_path), 0o755)
    
    gpu_name = gpu["name"]
    result = [
        "=== Fine-Tune Pipeline ===",
        f"Dataset: {ds_path}",
        f"Base model: {base_model}",
        f"GPU: {gpu_name}",
        "",
        f"Script: {script_path}",
        "",
        "Run it:",
        f"  python3 {script_path}",
        "",
        "Or with custom params:",
        f"  python3 {script_path} <model> <dataset> <output>",
        "",
        "Requirements:",
        "  pip install unsloth transformers datasets trl",
        "  NVIDIA GPU recommended (or Apple Silicon)",
    ]
    
    return "\n".join(result)
