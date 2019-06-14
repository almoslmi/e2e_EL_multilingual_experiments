from data_generator import load_data3, generate_wiki
from keras.layers import Input, Dense, Dropout, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


def build_model(sentdim=1024, entdim=100, gramdim=1024):
    in1 = Input(shape=(sentdim, ))
    in2 = Input(shape=(entdim, ))
    in3 = Input(shape=(gramdim, ))
    in4 = Input(shape=(1, ))
    x1 = Dense(1024, activation="relu")(in1)
    x2 = Dense(1024, activation="relu")(in2)
    x3 = Dense(1024, activation="relu")(in3)
    x4 = Dense(1, activation="relu")(in4)
    x = concatenate([x1, x2, x3])
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = concatenate([x, x4])
    out = Dense(1, activation="sigmoid")(x)
    model = Model([in1, in2, in3, in4], out)
    model.compile(optimizer="nadam",
                  loss="binary_crossentropy",
                  metrics=["acc"])
    return model


def run():
    pdata = load_data3()
    callbacks = [ModelCheckpoint("./model_wiki.h5")]
    model = build_model()
    model.fit_generator(generate_wiki(
        "../../../entity_types_scripts/ELdata_wiki.txt", *pdata),
                        steps_per_epoch=100,
                        epochs=1000,
                        callbacks=callbacks)


if __name__ == "__main__":
    run()
