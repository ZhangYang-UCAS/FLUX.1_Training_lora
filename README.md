使用flux训练自己的lora
## 数据准备
数据必须按照以下格式存储在自己的目录中：  
Your Image Directory  
├── img1.png  
├── img1.txt  
├── img2.png  
├── img2.txt  
...  
使用generate_lable.py生成图像的标签，将 (YOUR DIRECTORY NAME) 替换为您的图像文件夹的目录，其他内容无需更改。

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
使用以下终端命令登录，向 HuggingFace Cache 添加只读令牌：  
```
huggingface-cli login
```
