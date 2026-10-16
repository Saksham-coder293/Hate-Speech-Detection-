"""Gradio interface. Run standalone with `python demo.py`, or it gets
mounted into the FastAPI app (app.py) so one container serves both."""
import gradio as gr

from predict import predict_text

EXAMPLES = [
    "Have a wonderful day, my friend!",
    "go to hell you moron",
    "Those people are vermin and should be thrown out of our country.",
]

DISCLAIMER = (
    "**Prototype, not a moderation tool.** Trained on ~25k tweets (Davidson et al., 2017) "
    "with a TF-IDF + Logistic Regression model. Hate speech is the hardest class "
    "(precision ~0.35, recall ~0.60), the labels are noisy, and the data is known to "
    "over-flag African-American English. Treat output as a probability, not a verdict. "
    "Inputs are not stored."
)


def classify(text: str):
    if not text or not text.strip():
        raise gr.Error("Please enter some text.")
    try:
        return predict_text(text)["probabilities"]
    except ValueError as e:
        raise gr.Error(str(e))


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Hate Speech Detection") as demo:
        gr.Markdown("# Hate Speech Detection\nType a sentence and see how the model scores it.")
        inp = gr.Textbox(lines=4, max_length=1000, label="Text to analyze")
        btn = gr.Button("Analyze", variant="primary")
        out = gr.Label(num_top_classes=3, label="Class probabilities")
        gr.Examples(EXAMPLES, inputs=inp)
        btn.click(classify, inp, out, api_name=False)
        inp.submit(classify, inp, out, api_name=False)
        gr.Markdown(DISCLAIMER)
    return demo


if __name__ == "__main__":
    build_demo().launch()
