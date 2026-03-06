"""
测试脚本：验证模型梯度设置是否正确
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def test_gradient_setup():
    print("=" * 60)
    print("Testing Gradient Setup for GRPO Training")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n1. Loading model...")
    model_path = "../outputs/sft"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # 测试时使用CPU
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    print(f"✓ Model loaded: {type(model).__name__}")
    
    # 2. 准备模型
    print("\n2. Preparing model for training...")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True
    print("✓ Gradient checkpointing enabled")
    print("✓ Embeddings set to trainable")
    
    # 3. 应用LoRA
    print("\n3. Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, peft_config)
    print("✓ LoRA applied")
    
    # 4. 重新启用梯度
    print("\n4. Re-enabling gradients...")
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad_(True)
    
    # 获取嵌入层
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
    elif hasattr(model.base_model, "get_input_embeddings"):
        embeddings = model.base_model.get_input_embeddings()
    elif hasattr(model.base_model.model, "get_input_embeddings"):
        embeddings = model.base_model.model.get_input_embeddings()
    else:
        raise ValueError("Could not find input embeddings")
    
    for param in embeddings.parameters():
        param.requires_grad = True
    
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    
    embeddings.register_forward_hook(make_inputs_require_grad)
    print(f"✓ Hook registered on embeddings: {type(embeddings).__name__}")
    
    # 5. 验证设置
    print("\n5. Verifying setup...")
    print("-" * 60)
    
    # 检查可训练参数
    trainable_params = 0
    all_params = 0
    lora_params = 0
    embedding_params = 0
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if "lora" in name.lower():
                lora_params += param.numel()
            if "embed" in name.lower() or "lm_head" in name:
                embedding_params += param.numel()
    
    print(f"Total parameters: {all_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    print(f"  - LoRA parameters: {lora_params:,}")
    print(f"  - Embedding/LM head parameters: {embedding_params:,}")
    
    # 检查是否有任何可训练参数
    has_trainable = any(p.requires_grad for p in model.parameters())
    print(f"\nHas trainable parameters: {'✓ YES' if has_trainable else '✗ NO'}")
    
    # 测试前向传播
    print("\n6. Testing forward pass...")
    test_input = tokenizer("Test input", return_tensors="pt")
    
    try:
        with torch.enable_grad():
            outputs = model(**test_input)
            logits = outputs.logits
            
            # 检查输出是否需要梯度
            print(f"Output requires grad: {'✓ YES' if logits.requires_grad else '✗ NO'}")
            
            # 尝试计算一个简单的损失并反向传播
            loss = logits.mean()
            loss.backward()
            
            # 检查是否有梯度
            has_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    has_grad = True
                    break
            
            print(f"Gradients computed: {'✓ YES' if has_grad else '✗ NO'}")
            
            if has_grad:
                print("\n" + "=" * 60)
                print("✓✓✓ SUCCESS! Gradient setup is correct! ✓✓✓")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("✗✗✗ WARNING: No gradients found! ✗✗✗")
                print("=" * 60)
                
    except Exception as e:
        print(f"\n✗ Error during forward pass: {e}")
        print("=" * 60)
        raise

if __name__ == "__main__":
    test_gradient_setup()
