import tensorflow as tf


def model_fit(model, inp, out):
    model.fit(inp, out, batch_size=max(out.shape[0]//5, 1))


def save_model(model, filepath):
    json_data = model.to_json()
    with open(filepath + '.json', 'w') as json_file:
        json_file.write(json_data)
    model.save_weights(filepath + '.h5')


def load_model(filepath):
    json_data = None
    with open(filepath + '.json') as json_file:
        json_data = json_file.read()

    model = tf.keras.models.model_from_json(json_data)
    model.load_weights(filepath + '.h5')
    return model
