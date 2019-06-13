from data_generator import load_data, generate_test, gen_indices
from keras.models import load_model
from sklearn.metrics import classification_report


def run():
    svecs, labs, evecs = load_data()
    _, _, test_inds = gen_indices(len(labs))
    model = load_model("./models/model_best.h5")
    X_test, y_test = generate_test(test_inds, svecs, labs, evecs)
    y_preds = model.predict(X_test)
    y_preds = [y > 0.5 for y in y_preds]
    print(classification_report(y_test, y_preds))
    
    
if __name__ == "__main__":
    run()
