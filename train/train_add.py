from data_generator import load_data2, generate2, gen_indices
from keras.layers import Input, Dense, Dropout, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def run():
    pdata = load_data2()
    tinds, vinds, _ = gen_indices(len(pdata[1]))
    callbacks = [
        ModelCheckpoint("./model_aida_wiki_best.h5",
                        monitor="val_loss",
                        save_best_only=True,
                        mode=min)
    ]
    model = load_model("./model_wiki.h5")
    model.fit_generator(generate2(tinds, *pdata),
                        steps_per_epoch=300,
                        epochs=15,
                        callbacks=callbacks,
                        validation_data=generate2(vinds, *pdata),
                        validation_steps=1000)


if __name__ == "__main__":
    run()
