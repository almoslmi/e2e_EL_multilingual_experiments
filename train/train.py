from data_generator import load_data, generate, gen_indices
from keras.layers import Input, Dense, Dropout, Flatten, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint


def build_model(sentdim=1024, entdim=100):
    in1 = Input(shape=(1024, ))
    in2 = Input(shape=(100, ))
    x1 = Dense(1024, activation="relu")(in1)
    x2 = Dense(1024, activation="relu")(in2)
    x = concatenate([x1, x2])
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model([in1, in2], out)
    model.compile(optimizer="nadam",
                  loss="binary_crossentropy",
                  metrics=["acc"])
    return model


def run():
    svecs, labs, evecs = load_data()
    tinds, vinds, _ = gen_indices(len(labs))
    callbacks = [
        ModelCheckpoint("./model_best.h5",
                        monitor="val_loss",
                        save_best_only=True,
                        mode=min)
    ]
    model = build_model()
    model.fit_generator(generate(tinds, svecs, labs, evecs),
                        steps_per_epoch=12938,
                        epochs=3,
                        callbacks=callbacks,
                        validation_data=generate(vinds, svecs, labs, evecs),
                        validation_steps=1000)


if __name__ == "__main__":
    run()
