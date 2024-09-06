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

AI Toolkit 提供了一个训练脚本run.py, 可以处理训练 FLUX.1 模型的所有复杂问题。

您可以微调 schnell 或 dev 模型，但我们推荐使用 dev 模型。尽管 dev 的使用许可更加有限，但在快速理解、拼写和对象构成方面，它比 schnell 更加强大。不过，由于 schnell 更加精简，它的训练速度会更快。

run.py 通过 YAML 配置文件来处理训练参数。在这个例子中，我们将编辑 ```ai-toolkit/config/examples/train_lora_flux_24gb.yaml ```文件。

我们要编辑的最重要的几行是
1、第 5 行：更改模型名称。 
2、第 30 行：添加图像目录的路径。    
3、第 69 行和第 70 行：修改训练图像的高度和宽度   
4、第74-76行：编辑提示。如果您的目标是特定的人物、动物或场景，建议相应调整提示内容以更好地指导模型。您可以在提示中加入自定义关键词，以验证模型的训练效果。例如，若训练数据集为 anime style 风格图片，可以设置提示为```“anime style, a man holding a sign that says, ‘this is a sign’”```。   
5、第 80 和第 81 行的指导比例和采样步数来进一步控制生成样本。   

如果希望加快 FLUX.1 模型的训练速度，可以调整第 37 行的批处理大小和第 39 行的梯度累积步数。如果使用多 GPU 或 H100 进行训练，可以适当提高这些参数，但建议谨慎操作，以免导致内存不足错误。

在第 38 行，我们可以更改训练步骤数。他们建议在 500 到 4000 之间，所以我们选择中间值 2500。使用这个值我们得到了不错的结果。它会每 250 步检查一次，但如果需要，我们也可以在第 22 行更改这个值。

最后，我们可以将模型从HuggingFace中下载下来，并将路径粘贴到第 62 行（例如： “../black-forest-labs/FLUX.1-schnell”）。现在一切都已设置完毕，我们可以运行训练了！

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

