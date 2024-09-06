# 使用FLUX.1训练自己的lora

## 数据准备
数据必须按照以下格式存储在自己的目录中：  
Your Image Directory  
├── img1.png  
├── img1.txt  
├── img2.png  
├── img2.txt  
...  
使用generate_lable.py生成图像的标签，仅需将 ```<YOUR DIRECTORY NAME>``` 替换为您的图像文件夹的目录，其他内容无需更改。

基于[AI Toolkit](https://github.com/ostris/ai-toolkit)为基础开展这项工作

## 在终端中设置环境
首先获取以下代码并将其粘贴到终端中设置环境：
```
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
pip install peft
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 配置训练循环
AI Toolkit 提供了一个训练脚本，run.py可以处理训练 FLUX.1 模型的所有复杂问题。

可以对 schnell 或 dev 模型进行微调，但我们建议训练 dev 模型。dev 的使用许可更有限，但与 schnell 相比，它在快速理解、拼写和对象组成方面也更加强大。然而，由于 schnell 的提炼，它的训练速度应该要快得多。

run.py采用 yaml 配置文件来处理各种训练参数。对于此用例，我们将编辑该 ai-toolkit/config/examples/train_lora_flux_24gb.yaml 文件。

我们要编辑的最重要的几行是第 5 行 - 更改名称，第 30 行 - 添加图像目录的路径，以及第 69 行和第 70 行 - 我们可以编辑高度和宽度以反映我们的训练图像。编辑这些行以相应地调整训练程序以在您的图像上运行。

此外，我们可能想要编辑提示。一些提示涉及动物或场景，因此如果我们试图捕捉特定的人，我们可能想要编辑这些提示以更好地告知模型。在prompt里面可以添加自己的激活词,验证自己训练的是否有效。比如训练的数据集全都是anime风格图，就可以写一句"anime style, a man holding a sign that says, 'this is a sign'"，我们还可以使用第 80-81 行上的指导比例和样本步骤值进一步控制这些生成的样本。

如果我们想更快地训练 FLUX.1 模型，我们可以通过编辑批处理大小（第 37 行）和梯度累积步骤（第 39 行）来进一步优化模型训练。如果我们在多 GPU 或 H100 上进行训练，我们可以稍微提高这些值，但我们建议保持不变。请注意，提高这些值可能会导致内存不足错误。

在第 38 行，我们可以更改训练步骤数。他们建议在 500 到 4000 之间，所以我们选择中间值 2500。使用这个值我们得到了不错的结果。它会每 250 步检查一次，但如果需要，我们也可以在第 22 行更改这个值。

最后，我们可以将模型从HuggingFace中下载下来，粘贴到第 62 行（“../black-forest-labs/FLUX.1-schnell”）。现在一切都已设置完毕，我们可以运行训练了！

## 运行 FLUX.1 训练循环

要运行训练循环，我们现在需要做的就是使用脚本run.py。
```
 python3 run.py config/examples/train_lora_flux_24gb.yaml
```
## 使用 FLUX.1 LoRA 进行推理
```
import torch
from diffusers import FluxPipeline

model_id = '../black-forest-labs/FLUX.1-dev'
ckpt_name = f'{lora_name}.safetensors'

pipeline = FluxPipeline.from_pretrained(model_id)
pipeline.load_lora_weights(ckpt_name)
pipeline.to('cuda', dtype=torch.float16)

prompt = "a photo of a cat"

image = pipeline(
    prompt,
    width=1024,
    height=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("output.png")
```

