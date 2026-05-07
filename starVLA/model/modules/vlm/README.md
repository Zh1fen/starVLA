# VLM 模块设计 (`starVLA/model/modules/vlm/`)

## 定位

VLM 模块封装各种 **Vision-Language Model** 后端（Qwen2.5-VL、Qwen3-VL、Qwen3.5-VL、Florence-2），
为上层 `framework/VLM4A/` 中的 VLA 框架提供统一接口。

## 接口规范

每个 VLM wrapper 都继承 `nn.Module`，实现以下方法：

| 方法 | 签名 | 用途 |
|------|------|------|
| `__init__` | `(config)` | 加载预训练模型 + processor |
| `forward` | `(**kwargs) -> CausalLMOutputWithPast` | 前向传播（训练/推理） |
| `generate` | `(**kwargs)` | 自回归生成 |
| `build_qwenvl_inputs` | `(images, instructions, solutions=None)` | 构建模型输入 |

## 工厂函数

```python
from starVLA.model.modules.vlm import get_vlm_model
vlm = get_vlm_model(config)  # 根据 config.framework.qwenvl.base_vlm 路由
```

## 现有实现

| 文件 | 类名 | 支持的模型 |
|------|------|-----------|
| `QWen2_5.py` | `_QWen_VL_Interface` | Qwen2.5-VL 系列 |
| `QWen3.py` | `_QWen3_VL_Interface` | Qwen3-VL 系列 |
| `QWen3_5.py` | `_QWen3_5_VL_Interface` | Qwen3.5-VL 系列 |
| `Florence2.py` | `_Florence_Interface` | Florence-2 系列 |

## Qwen LoRA 微调

第一版 LoRA 只接在 Qwen 系 wrapper 上：

- `QWen2_5.py`
- `QWen3.py`
- `QWen3_5.py`

配置入口统一放在 `framework.qwenvl` 下：

```yaml
framework:
  qwenvl:
    enable_gradient_checkpointing: true
    lora:
      enabled: true
      r: 16
      alpha: 32
      dropout: 0.05
      bias: none
      target_modules: auto
```

注意：

- `trainer.gradient_checkpointing` 不是 VLM 的真实开关；Qwen wrapper 实际读取的是 `framework.qwenvl.enable_gradient_checkpointing`
- 开启 LoRA 时，不要再把 `qwen_vl_interface` 写进 `trainer.freeze_modules`
- `learning_rate.qwen_vl_interface` 会自动只作用于 trainable 的 LoRA 参数，而不是整套 base VLM

## 与 World Model 的关系

Cosmos-Reason2 已迁移至 `starVLA/model/modules/world_model/`。
如果 `config.framework.qwenvl.base_vlm` 包含 `cosmos-reason2`，
`get_vlm_model()` 会自动委托给 `get_world_model()`，保持向后兼容。

## 数据流

```
Framework.__init__()
  └─> get_vlm_model(config) -> VLM wrapper instance
       └─> self.qwen_vl_interface

Framework.forward(examples)
  └─> self.qwen_vl_interface.build_qwenvl_inputs(images, instructions)
       └─> BatchFeature {input_ids, attention_mask, pixel_values, ...}
  └─> self.qwen_vl_interface(**inputs, output_hidden_states=True)
       └─> hidden_states[-1] : [B, L, H]
  └─> action_model(hidden_states, actions) -> loss
```
