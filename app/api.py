from flask import Flask, jsonify, request
import os
os.environ["ONEDNN_VERBOSE"] = "0"
os.environ["DNNL_VERBOSE"] = "0"
os.environ["ATEN_CPU_CAPABILITY"] = "default"
import torch
import os

app = Flask(__name__)

# Load TorchScript model
MODEL_PATH = os.path.join("trained_model", "deep_mlp_api.pt")

model = torch.jit.load(MODEL_PATH)
model.eval()

@app.route("/")
def home():
    return jsonify({
        "message": "Lung Nodule Detection API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = data.get("features")

    if features is None:
        return jsonify({
            "status": "error",
            "message": "Please send features list"
        })

    if len(features) != 8:
        return jsonify({
            "status": "error",
            "message": "Model expects 8 input features"
        })

    input_tensor = torch.tensor([features], dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()

    result = "Malignant" if prediction == 1 else "Benign"

    return jsonify({
        "status": "success",
        "prediction_class": prediction,
        "prediction_result": result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)