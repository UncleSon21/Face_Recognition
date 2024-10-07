import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from model import siamese_model
from preprocess_images import create_dataset
from config import EPOCHS, LEARNING_RATE, CHECKPOINT_PREFIX, MODEL_PATH

@tf.function
def train_step(batch, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]
        yhat = model(X, training=True)
        loss = loss_fn(y, yhat)
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    return loss

def train():
    train_data, _ = create_dataset()
    model = siamese_model()
    binary_cross_loss = tf.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(LEARNING_RATE)

    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=model)

    for epoch in range(1, EPOCHS + 1):
        print(f'\n Epoch {epoch}/{EPOCHS}')
        progbar = tf.keras.utils.Progbar(len(train_data))
        r = Recall()
        p = Precision()

        for idx, batch in enumerate(train_data):
            loss = train_step(batch, model, binary_cross_loss, opt)
            yhat = model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx + 1)

        print(f"Loss: {loss.numpy()}, Recall: {r.result().numpy()}, Precision: {p.result().numpy()}")

        if epoch % 10 == 0:
            checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()