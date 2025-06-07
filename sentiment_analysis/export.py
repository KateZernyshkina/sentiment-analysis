import tensorflow as tf


def export_to_onnx(model_path="model.keras", export_path="model.onnx"):
    import tf2onnx

    model = tf.keras.models.load_model(model_path)
    spec = (tf.TensorSpec((None, 150), tf.int32, name="input"),)
    output_path = export_path
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    export_to_onnx()
