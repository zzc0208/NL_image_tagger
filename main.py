import os
import gradio as gr
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from tqdm import tqdm
from io import StringIO

# 加载模型和tokenizer
model_name_or_path = "models/model"  # 使用你的模型路径
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

def process_image(image_path, question, top_k=50, top_p=0.9, temperature=0.7):
    image = Image.open(image_path).convert('RGB')
    msgs = [{'role': 'user', 'content': [image, question]}]

    result = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature
    )
    return result

def describe_images(directory, temperature):
    log = ""
    question = "Describe this picture"
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 创建一个字符串IO对象，用于捕获tqdm的输出
    tqdm_out = StringIO()

    with tqdm(total=len(image_files), desc="Processing Images", file=tqdm_out, leave=False) as pbar:
        for filename in image_files:
            filepath = os.path.join(directory, filename)
            
            # 生成图像描述
            description = process_image(filepath, question, temperature=temperature)
            
            # 保存描述为txt文件
            text_filename = os.path.splitext(filename)[0] + ".txt"
            text_filepath = os.path.join(directory, text_filename)
            with open(text_filepath, 'w') as text_file:
                text_file.write(description)
            
            # 更新log
            log += f"Processed {filename}, saved description to {text_filename}\n"
            pbar.update(1)

            # 重置tqdm_out并获取最新的进度条状态
            tqdm_out.seek(0)
            tqdm_out.truncate(0)
            pbar.display()  # 刷新进度条状态到tqdm_out
            tqdm_out.seek(0)
            tqdm_log = tqdm_out.read().strip().split('\n')[-1]  # 只获取最后一行

            # 将进度条信息添加到日志中
            full_log = log + tqdm_log + '\n'
            
            # 实时返回更新后的log
            yield full_log

# Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("## NL_image_tagger")
    gr.Markdown("#### 此项目基于多模态大语言模型对图片进行自然语言打标")
    gr.Markdown("#### 输入需要打标的图片目录，并点击开始处理即可")
    directory_input = gr.Textbox(label="图片目录", placeholder="输入图片所在的文件夹路径")
    temperature_input = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.7, label="Temperature (一般情况下无需调整)")
    result_box = gr.Textbox(label="日志", placeholder="处理日志将显示在这里", lines=10)
    
    run_button = gr.Button("开始处理")
    
    def process_images(directory, temperature):
        log_gen = describe_images(directory, temperature)
        for log in log_gen:
            yield gr.update(value=log)
    
    run_button.click(fn=process_images, inputs=[directory_input, temperature_input], outputs=result_box)

demo.launch()
