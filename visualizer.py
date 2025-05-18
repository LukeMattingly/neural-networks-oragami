import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import anywidget
    import traitlets
    import numpy as np
    import json
    from pathlib import Path

    from model import model_forward, load_weights
    return Path, anywidget, load_weights, mo, model_forward, np, traitlets


@app.cell(hide_code=True)
def _(anywidget, np, traitlets):
    class DrawWidget(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
          const canvas = document.createElement("canvas");
          canvas.width = 28;
          canvas.height = 28;
          canvas.style.width = "280px";
          canvas.style.height = "280px";
          canvas.style.border = "1px solid #ccc";
          canvas.style.imageRendering = "pixelated";
          el.appendChild(canvas);

          const ctx = canvas.getContext("2d");
          ctx.fillStyle = "black";
          ctx.fillRect(0, 0, canvas.width, canvas.height);

          let drawing = false;
          let eraseMode = false;

          function draw(e) {
            if (!drawing) return;
            const rect = canvas.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) * (canvas.width / canvas.offsetWidth));
            const y = Math.floor((e.clientY - rect.top) * (canvas.height / canvas.offsetHeight));
            ctx.fillStyle = eraseMode ? "black" : "white";
            ctx.fillRect(x, y, 2, 2);
            updateModel();
          }

          canvas.addEventListener("mousedown", () => { drawing = true; });
          canvas.addEventListener("mouseup", () => { drawing = false; });
          canvas.addEventListener("mouseleave", () => { drawing = false; });
          canvas.addEventListener("mousemove", draw);

          const controls = document.createElement("div");
          controls.style.marginTop = "10px";

          const toggleButton = document.createElement("button");
          toggleButton.textContent = "Switch to Erase";
          toggleButton.onclick = () => {
            eraseMode = !eraseMode;
            toggleButton.textContent = eraseMode ? "Switch to Draw" : "Switch to Erase";
          };
          controls.appendChild(toggleButton);

          const clearButton = document.createElement("button");
          clearButton.textContent = "Clear";
          clearButton.style.marginLeft = "10px";
          clearButton.onclick = () => {
            ctx.fillStyle = "black";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            updateModel();
          };
          controls.appendChild(clearButton);

          el.appendChild(controls);

          function updateModel() {
            // Only grayscale: one value per pixel
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            // Grayscale value: just use the R channel (since we only draw in black and white)
            const data = [];
            for (let i = 0; i < imageData.data.length; i += 4) {
              // 0 (black) to 255 (white)
              data.push(imageData.data[i]);
            }
            model.set("image_data", data);
            model.save_changes();
          }

          // Set default blank (all black)
          updateModel();

          model.on("change:image_data", () => {
            // No need to update canvas from model (1-way sync is sufficient)
          });
        }
        export default { render };
        """
        image_data = traitlets.List(
            trait=traitlets.Int(), default_value=[0] * 28 * 28
        ).tag(sync=True)

        @property
        def numpy(self):
            return np.array(self.image_data, dtype=np.uint8)
    return (DrawWidget,)


@app.cell
def _(DrawWidget, mo):
    canvas = mo.ui.anywidget(DrawWidget())
    return (canvas,)


@app.cell
def _(Path, load_weights):
    weights = load_weights(Path("weights.pkl"))
    return (weights,)


@app.cell
def _(weights):
    weights
    return


@app.cell
def _(canvas, model_forward, weights):
    pred = model_forward(weights, canvas.numpy / 255.0).argmax().item()
    return (pred,)


@app.cell
def _(canvas, mo, pred):
    mo.hstack(
        [canvas, mo.md(f"# Prediction: {pred}")],
        align="center",
        justify="center",
        gap=10,
    )
    return


if __name__ == "__main__":
    app.run()
