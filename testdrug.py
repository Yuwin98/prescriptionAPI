import keras
from util import prepare_image_for_prediction, decode_batch_predictions


def test_drug():
    trained_model = keras.models.load_model("./models/drugs_v2.h5")
    drug_predictor = keras.models.Model(
        trained_model.get_layer(name="image").input, trained_model.get_layer(name="dense2").output
    )
    drug_path = "static/img_07c49cc4-3b95-11ed-9490-2016b9895a07/drug.jpeg"
    drug_img = prepare_image_for_prediction([drug_path], [""])

    preds = drug_predictor.predict(drug_img)
    drug = decode_batch_predictions(preds)
    print(drug)


test_drug()
