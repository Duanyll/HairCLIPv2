from inference import InferenceProxy
from PIL import ImageColor
import gradio as gr

def main():
    proxy = InferenceProxy()
    print(">>> Waiting Gradio to start...")
    def edit_by_text(image_path, text_cond):
        return proxy.edit_by_text(image_path, text_cond)
    def edit_by_ref(image_path, ref_path):
        return proxy.edit_by_ref(image_path, ref_path)
    def edit_color_by_ref(image_path, ref_path):
        return proxy.edit_color(image_path, ref_path)
    def edit_color_by_value(image_path, color_cond):
        # Parse HTML color to RGB tuple
        if color_cond is None:
            color_cond = "#000000"
        color_cond = ImageColor.getcolor(color_cond, "RGB")
        return proxy.edit_color(image_path, color_cond)
    app = gr.TabbedInterface([
        gr.Interface(edit_by_text, 
                     inputs=[gr.Image(type="pil", label="Image"), gr.Textbox(label="Text")],
                     outputs="image",
                     api_name="edit_by_text"
        ),
        gr.Interface(edit_by_ref, 
                     inputs=[gr.Image(type="pil", label="Image"), gr.Image(type="pil", label="Reference")],
                     outputs="image",
                     api_name="edit_by_ref"
        ),
        gr.Interface(edit_color_by_ref, 
                     inputs=[gr.Image(type="pil", label="Image"), gr.Image(type="pil", label="Reference")],
                     outputs="image",
                     api_name="edit_color_by_ref"
        ),
        gr.Interface(edit_color_by_value, 
                     inputs=[gr.Image(type="pil", label="Image"), gr.ColorPicker(label="Color")],
                     outputs="image",
                     api_name="edit_color_by_value"
        ),
    ], tab_names=[
        "Text", "Reference", "Color by Reference", "Color by Value"
    ], title="Hairstyle Editor")
    
    app.launch()
    
if __name__ == "__main__":
    main()