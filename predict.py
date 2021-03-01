# takes a single image as input
# gives a string as output
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2

classes = {
    "alai_darwaza": 0,
    "alai_minar": 1,
    "iron_pillar": 2,
    "jamali_kamali_tomb": 3,
    "qutub_minar": 4,
}

app = Flask(__name__)


def processing(img_2):

    model = load_model("SLR_bw_black4.h5")

    img_path = "test_images/"

    img = cv2.imread(img_2)

    rszd_img = cv2.resize(img, (128, 96))

    rszd_img = rszd_img.reshape(1, 96, 128, 3)

    rszd_img = rszd_img / 255

    prediction = model.predict_classes(rszd_img)

    reslt_class = list(classes.keys())[prediction[0]]

    return reslt_class


@app.route("/upload", methods=["GET", "POST"])
def upload():
    file = request.files["image"]
    file.save("./image.jpg")
    img_2 = "./image.jpg"
    predictions = processing(img_2)
    return jsonify(
        {
            "result": predictions,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
