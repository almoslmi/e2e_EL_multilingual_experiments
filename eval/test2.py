from data_generator import load_data2, generate2_test, gen_indices
from keras.models import load_model
from sklearn.metrics import classification_report


def run():
    pdata = load_data2()
    _, _, test_inds = gen_indices(len(pdata[1]))
    model = load_model("./model2_best.h5")
    X_test, y_test = generate2_test(test_inds, *pdata)
    y_preds = model.predict(X_test)
    y_preds = [y > 0.5 for y in y_preds]
    print(classification_report(y_test, y_preds))
    
    
if __name__ == "__main__":
    run()
