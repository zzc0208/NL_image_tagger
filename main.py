import os
import gradio as gr
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from tqdm import tqdm

# 加载模型和tokenizer
model_name_or_path = "models/model"  # 使用你的模型路径
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

def process_image(image_path, question, top_k=50, top_p=0.9, temperature=0.7):
    try:
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
    except Exception as e:
        return f"Error processing image {image_path}: {e}"

def describe_images(directory, temperature):
    log = ""
    question = "Describe this picture"
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not os.path.isdir(directory):
        yield "错误: 输入的图片目录不存在！"
        return

    with tqdm(total=len(image_files), desc="Processing Images") as pbar:
        for filename in image_files:
            filepath = os.path.join(directory, filename)

            try:
                description = process_image(filepath, question, temperature=temperature)
                if description.startswith("Error"):  # 检查 process_image 是否返回错误信息
                    log += f"Failed to process {filename}: {description.split(': ', 1)[1]}\n"
                else:
                    # 保存描述为txt文件
                    text_filename = os.path.splitext(filename)[0] + ".txt"
                    text_filepath = os.path.join(directory, text_filename)
                    with open(text_filepath, 'w', encoding='utf-8') as text_file:
                        text_file.write(description)
                    log += f"Processed {filename}, saved description to {text_filename}\n"
            except Exception as e:
                log += f"Error during processing {filename}: {e}\n"

            pbar.update(1)
            yield log  # 实时 yield 更新后的日志

    yield log + "\n图片描述处理完成！" # 处理完成后添加提示信息

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
        result_box_update = gr.Textbox(value="", label="日志", placeholder="处理日志将显示在这里", lines=10) # 每次开始处理时清空日志框
        yield result_box_update
        log_gen = describe_images(directory, temperature)
        for log in log_gen:
            result_box_update = gr.Textbox(value=log, label="日志", placeholder="处理日志将显示在这里", lines=10)
            yield result_box_update


    run_button.click(fn=process_images, inputs=[directory_input, temperature_input], outputs=result_box)

demo.launch()
