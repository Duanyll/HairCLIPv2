import gradio as gr
from inference import InferenceProxy

def main():
    proxy = InferenceProxy()
    def edit_by_text(image_path, text_cond):
        return proxy.edit_by_text(image_path, text_cond)
    def edit_by_ref(image_path, ref_path):
        return proxy.edit_by_ref(image_path, ref_path)
    def edit_color_by_ref(image_path, ref_path):
        return proxy.edit_color(image_path, ref_path)
    def edit_color_by_value(image_path, color_cond):
        # Parse HTML color to RGB tuple
        color_cond = tuple(int(color_cond[i:i+2], 16) for i in (1, 3, 5))
        return proxy.edit_color(image_path, color_cond)
    app = gr.TabbedInterface([
        gr.Interface(edit_by_text, 
                     inputs=[gr.Image(type="filepath", label="Image"), gr.Textbox(label="Text")],
                     outputs="image",
                     api_name="edit_by_text"
        ),
        gr.Interface(edit_by_ref, 
                     inputs=[gr.Image(type="filepath", label="Image"), gr.Image(type="filepath", label="Reference")],
                     outputs="image",
                     api_name="edit_by_ref"
        ),
        gr.Interface(edit_color_by_ref, 
                     inputs=[gr.Image(type="filepath", label="Image"), gr.Image(type="filepath", label="Reference")],
                     outputs="image",
                     api_name="edit_color_by_ref"
        ),
        gr.Interface(edit_color_by_value, 
                     inputs=[gr.Image(type="filepath", label="Image"), gr.ColorPicker(label="Color")],
                     outputs="image",
                     api_name="edit_color_by_value"
        ),
    ], tab_names=[
        "Text", "Reference", "Color by Reference", "Color by Value"
    ], title="Hairstyle Editor")
    
    app.launch()
    
if __name__ == "__main__":
    main()