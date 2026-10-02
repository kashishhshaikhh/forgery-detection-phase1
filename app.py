import os
import uuid

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from casia_predict import predict_casia
from cifake_predict import predict_cifake


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}

MAX_UPLOAD_MB = 10

# Display bands only.
# These are NOT changing the actual model predictions.
LOW = 0.30
HIGH = 0.70


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_UPLOAD_MB"] = MAX_UPLOAD_MB

app.config["MAX_CONTENT_LENGTH"] = (
    MAX_UPLOAD_MB * 1024 * 1024
)


def allowed_file(filename):

    extension = os.path.splitext(
        filename
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


def display_casia_label(score):

    if score >= HIGH:
        return "TAMPERED"

    if score <= LOW:
        return "REAL"

    return "INCONCLUSIVE"


def display_cifake_label(score):

    if score >= HIGH:
        return "REAL"

    if score <= LOW:
        return "AI-GENERATED"

    return "INCONCLUSIVE"


@app.route("/", methods=["GET", "POST"])
def home():

    context = {
        "final_result": None,
        "final_score": None,

        "casia_result": None,
        "casia_score": None,

        "cifake_result": None,
        "cifake_score": None,

        "image_file": None,
        "result_comment": None,
    }

    if request.method == "POST":

        file = request.files.get("image")

        if file is None or file.filename == "":
            return render_template(
                "index.html",
                error="Please select an image."
            )

        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                error=(
                    "Unsupported file type. "
                    "Please upload a JPG or PNG image."
                )
            )

        filename = (
            f"{uuid.uuid4().hex[:8]}_"
            f"{secure_filename(file.filename)}"
        )

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(image_path)

        context["image_file"] = filename

        try:

            # --------------------------------
            # RUN CASIA
            # --------------------------------

            casia_model_result, casia_score = (
                predict_casia(image_path)
            )

            # --------------------------------
            # RUN CIFAKE
            # --------------------------------

            cifake_model_result, cifake_score = (
                predict_cifake(image_path)
            )

            # --------------------------------
            # DISPLAY LABELS
            # --------------------------------

            casia_result = display_casia_label(
                casia_score
            )

            cifake_result = display_cifake_label(
                cifake_score
            )

            print(
                f"[RESULT] {filename}"
                f" | CASIA: "
                f"{casia_model_result}"
                f" ({casia_score:.4f})"
                f" | CIFAKE: "
                f"{cifake_model_result}"
                f" ({cifake_score:.4f})"
            )

            context.update(
                casia_result=casia_result,
                casia_score=casia_score,

                cifake_result=cifake_result,
                cifake_score=cifake_score,
            )

            # --------------------------------
            # FINAL RESULT
            # --------------------------------

            if casia_result == "TAMPERED":

                context["final_result"] = "TAMPERED"

                context["final_score"] = casia_score

                context["result_comment"] = (
                    "The image shows patterns that "
                    "may indicate editing or "
                    "tampering according to the "
                    "ELA-based detector."
                )

            elif cifake_result == "AI-GENERATED":

                context["final_result"] = (
                    "AI-GENERATED"
                )

                context["final_score"] = cifake_score

                context["result_comment"] = (
                    "The AI-generation detector "
                    "found patterns commonly "
                    "associated with synthetic "
                    "images."
                )

            elif (
                casia_result == "INCONCLUSIVE"
                or
                cifake_result == "INCONCLUSIVE"
            ):

                context["final_result"] = (
                    "INCONCLUSIVE"
                )

                context["final_score"] = (
                    casia_score
                )

                context["result_comment"] = (
                    "At least one detector produced "
                    "an uncertain result. The "
                    "individual model outputs are "
                    "shown below."
                )

            else:

                context["final_result"] = "REAL"

                context["final_score"] = (
                    cifake_score
                )

                context["result_comment"] = (
                    "Neither detector flagged the "
                    "image. This is a model prediction "
                    "and not definitive forensic proof."
                )

        except Exception as e:

            return render_template(
                "index.html",
                error=(
                    f"Error while analyzing image: {e}"
                )
            )

    return render_template(
        "index.html",
        **context
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )