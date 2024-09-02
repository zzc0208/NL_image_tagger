# NL_image_tagger

### NL_image_tagger 是一个基于本地多模态大语言模型对图片进行批量自然语言打标的项目

---

## 不同模型的配置需求

本项目支持多种模型，欢迎各位使用不同的模型测试效果以及配置需求，并将测试结果反馈到 [issue](#) 中，我会将您提供的数据加入 README，以下是几种示例模型：

- **[openbmb/MiniCPM-V-2_6-int4](https://huggingface.co/openbmb/MiniCPM-V-2_6-int4)**  
  - 需求：10G VRAM  
  - 测试结果：此模型在使用 RTX 4070 SUPER 12G 时的打标速度约为 10 秒/张  
  - 测试者：[zzc0208](https://github.com/zzc0208)

- **[OpenGVLab/InternVL2-26B-AWQ](https://huggingface.co/OpenGVLab/InternVL2-26B-AWQ)**

- **[OpenGVLab/InternVL2-40B-AWQ](https://huggingface.co/OpenGVLab/InternVL2-40B-AWQ)**

---

## 使用教学

### 1. 安装依赖

推荐使用 `Python 3.10.11`

#### 1.1. 安装 PyTorch

在 [pytorch.org](https://pytorch.org/get-started/locally/) 获取适合自己的安装命令，示例如下：

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade
```

#### 1.2. 安装其余依赖

```shell
pip install huggingface_hub gradio transformers pillow tqdm --upgrade
```

### 2. 下载模型

（可选：更换模型）修改 `downloadmodel.py` 文件中 `line 3` 的 `repo_id="openbmb/MiniCPM-V-2_6-int4"`

```shell
python downloadmodel.py
```

### 3. 启动主程序

```shell
python main.py
```

### 4.使用主程序进行打标
在`图片目录`处输入需要打标的图片目录，并点击`开始处理`，程序就会自动开始标注

![image](https://github.com/user-attachments/assets/f72af810-60b4-4ce8-b2fd-dff6dc90646d)
![image](https://github.com/user-attachments/assets/4296a0d8-f42e-4b63-a9e3-2fd3f181a69a)
![image](https://github.com/user-attachments/assets/b2cc6db7-f731-4b43-987b-5fe62c5894bb)

