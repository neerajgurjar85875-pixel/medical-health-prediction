from flask import Flask, render_template, request
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime


app = Flask(__name__)


# =========================================================
# LOAD TRAINED ML MODEL
# =========================================================

model = joblib.load("model.pkl")


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# DASHBOARD PAGE
# =========================================================

@app.route("/dashboard")
def dashboard():

    # -----------------------------------------------------
    # Load Original Dataset
    # -----------------------------------------------------

    df = pd.read_csv("insurance(in).csv")


    # -----------------------------------------------------
    # Dataset Statistics
    # -----------------------------------------------------

    total_customers = len(df)


    average_charges = round(
        df["charges"].mean(),
        2
    )


    average_bmi = round(
        df["bmi"].mean(),
        2
    )


    average_age = round(
        df["age"].mean(),
        2
    )


    # -----------------------------------------------------
    # Smoker Percentage
    # -----------------------------------------------------

    smoker_counts = df["smoker"].value_counts(
        normalize=True
    )

    smoker_percentage = round(
        smoker_counts.get("yes", 0) * 100,
        2
    )


    # =====================================================
    # PREDICTION HISTORY STATISTICS
    # =====================================================

    history_file = "prediction_history.csv"


    # Default values
    total_predictions = 0
    average_prediction = 0
    highest_prediction = 0
    lowest_prediction = 0

    smoker_prediction = 0
    non_smoker_prediction = 0


    # -----------------------------------------------------
    # Check Prediction History File
    # -----------------------------------------------------

    if os.path.exists(history_file):

        history_data = pd.read_csv(
            history_file
        )


        # -------------------------------------------------
        # Check History Data
        # -------------------------------------------------

        if not history_data.empty:

            # Total predictions
            total_predictions = len(
                history_data
            )


            # Average predicted cost
            average_prediction = round(
                history_data["Predicted Cost"].mean(),
                2
            )


            # Highest predicted cost
            highest_prediction = round(
                history_data["Predicted Cost"].max(),
                2
            )


            # Lowest predicted cost
            lowest_prediction = round(
                history_data["Predicted Cost"].min(),
                2
            )


            # -------------------------------------------------
            # Smoker Prediction
            # -------------------------------------------------

            smoker_data = history_data[
                history_data["Smoker"].astype(str).str.lower() == "yes"
            ]


            if not smoker_data.empty:

                smoker_prediction = round(
                    smoker_data["Predicted Cost"].mean(),
                    2
                )


            # -------------------------------------------------
            # Non-Smoker Prediction
            # -------------------------------------------------

            non_smoker_data = history_data[
                history_data["Smoker"].astype(str).str.lower() == "no"
            ]


            if not non_smoker_data.empty:

                non_smoker_prediction = round(
                    non_smoker_data["Predicted Cost"].mean(),
                    2
                )


    # =====================================================
    # SEND DATA TO DASHBOARD
    # =====================================================

    return render_template(

        "dashboard.html",

        # Dataset statistics
        total_customers=total_customers,
        average_charges=average_charges,
        average_bmi=average_bmi,
        average_age=average_age,
        smoker_percentage=smoker_percentage,

        # Prediction statistics
        total_predictions=total_predictions,
        average_prediction=average_prediction,
        highest_prediction=highest_prediction,
        lowest_prediction=lowest_prediction,

        # Smoker statistics
        smoker_prediction=smoker_prediction,
        non_smoker_prediction=non_smoker_prediction
    )


# =========================================================
# PREDICTION PAGE
# =========================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():


    # =====================================================
    # GET REQUEST
    # =====================================================

    if request.method == "GET":

        return render_template(
            "predict.html"
        )


    # =====================================================
    # POST REQUEST
    # =====================================================

    try:

        # -------------------------------------------------
        # Get Input Values
        # -------------------------------------------------

        age = int(
            request.form["age"]
        )


        sex = request.form["sex"]


        bmi = float(
            request.form["bmi"]
        )


        children = int(
            request.form["children"]
        )


        smoker = request.form["smoker"]


        region = request.form["region"]


        # =================================================
        # INPUT VALIDATION
        # =================================================

        # Age validation
        if age < 1 or age > 100:

            return render_template(
                "predict.html",
                error="Please enter a valid age between 1 and 100."
            )


        # BMI validation
        if bmi < 10 or bmi > 60:

            return render_template(
                "predict.html",
                error="Please enter a valid BMI between 10 and 60."
            )


        # Children validation
        if children < 0 or children > 20:

            return render_template(
                "predict.html",
                error="Please enter a valid number of children between 0 and 20."
            )


        # =================================================
        # CREATE INPUT DATAFRAME
        # =================================================

        input_data = pd.DataFrame({

            "age": [age],

            "sex": [sex],

            "bmi": [bmi],

            "children": [children],

            "smoker": [smoker],

            "region": [region]

        })


        # =================================================
        # ML PREDICTION
        # =================================================

        prediction = model.predict(
            input_data
        )[0]


        prediction = round(
            float(prediction),
            2
        )


        # =================================================
        # SAVE PREDICTION HISTORY
        # =================================================

        history_file = "prediction_history.csv"


        history_data = pd.DataFrame([{

            "Date & Time":
                datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),

            "Age":
                age,

            "Sex":
                sex,

            "BMI":
                bmi,

            "Children":
                children,

            "Smoker":
                smoker,

            "Region":
                region,

            "Predicted Cost":
                prediction

        }])


        # -------------------------------------------------
        # If history file exists
        # -------------------------------------------------

        if os.path.exists(history_file):

            history_data.to_csv(

                history_file,

                mode="a",

                header=False,

                index=False
            )


        # -------------------------------------------------
        # If history file does not exist
        # -------------------------------------------------

        else:

            history_data.to_csv(

                history_file,

                mode="w",

                header=True,

                index=False
            )


        # =================================================
        # PREDICTION EXPLANATION
        # =================================================

        factors = []


        # -------------------------------------------------
        # Smoking Factor
        # -------------------------------------------------

        if smoker == "yes":

            factors.append(
                "Smoking status can significantly increase "
                "the estimated insurance cost."
            )

        else:

            factors.append(
                "Non-smoking status generally contributes "
                "to a lower estimated cost."
            )


        # -------------------------------------------------
        # BMI Factor
        # -------------------------------------------------

        if bmi >= 30:

            factors.append(
                "BMI is in the higher range and may contribute "
                "to increased medical expenses."
            )


        elif bmi >= 25:

            factors.append(
                "BMI is in the overweight range and can influence "
                "medical insurance cost."
            )


        else:

            factors.append(
                "BMI is below the overweight range."
            )


        # -------------------------------------------------
        # Age Factor
        # -------------------------------------------------

        if age >= 50:

            factors.append(
                "Higher age can be associated with increased "
                "healthcare expenses."
            )


        elif age >= 30:

            factors.append(
                "Age is an important demographic factor "
                "considered by the model."
            )


        else:

            factors.append(
                "The entered age is relatively young."
            )


        # -------------------------------------------------
        # Children Factor
        # -------------------------------------------------

        if children > 2:

            factors.append(
                "The number of dependents may also influence "
                "insurance-related costs."
            )


        # =================================================
        # INPUT SUMMARY
        # =================================================

        summary = {

            "Age":
                age,

            "Sex":
                sex,

            "BMI":
                bmi,

            "Children":
                children,

            "Smoker":
                smoker,

            "Region":
                region
        }


        # =================================================
        # RESULT PAGE
        # =================================================

        return render_template(

            "result.html",

            prediction=prediction,

            factors=factors,

            summary=summary
        )


    # =====================================================
    # VALUE ERROR
    # =====================================================

    except ValueError:

        return render_template(

            "predict.html",

            error="Please enter valid numeric values."
        )


    # =====================================================
    # OTHER ERRORS
    # =====================================================

    except Exception as e:

        print(
            "Prediction Error:",
            e
        )

        return render_template(

            "predict.html",

            error="Something went wrong. Please check your inputs."
        )


# =========================================================
# PREDICTION HISTORY PAGE
# =========================================================

@app.route("/history")
def history():


    history_file = "prediction_history.csv"


    # -----------------------------------------------------
    # Check History File
    # -----------------------------------------------------

    if os.path.exists(history_file):

        history_data = pd.read_csv(
            history_file
        )


        # -------------------------------------------------
        # Convert DataFrame to Dictionary
        # -------------------------------------------------

        history_records = history_data.to_dict(
            orient="records"
        )


    else:

        history_records = []


    # -----------------------------------------------------
    # Send History to HTML
    # -----------------------------------------------------

    return render_template(

        "history.html",

        history=history_records
    )


# =========================================================
# DATA ANALYSIS PAGE
# =========================================================

@app.route("/analysis")
def analysis():


    # -----------------------------------------------------
    # Load Dataset
    # -----------------------------------------------------

    df = pd.read_csv(
        "insurance(in).csv"
    )


    # -----------------------------------------------------
    # Create Static Folder
    # -----------------------------------------------------

    os.makedirs(
        "static",
        exist_ok=True
    )


    # =====================================================
    # 1. AGE DISTRIBUTION
    # =====================================================

    plt.figure(
        figsize=(8, 5)
    )


    plt.hist(
        df["age"],
        bins=15
    )


    plt.xlabel(
        "Age"
    )


    plt.ylabel(
        "Number of Customers"
    )


    plt.title(
        "Age Distribution"
    )


    plt.tight_layout()


    plt.savefig(
        "static/age_distribution.png"
    )


    plt.close()


    # =====================================================
    # 2. BMI DISTRIBUTION
    # =====================================================

    plt.figure(
        figsize=(8, 5)
    )


    plt.hist(
        df["bmi"],
        bins=20
    )


    plt.xlabel(
        "BMI"
    )


    plt.ylabel(
        "Number of Customers"
    )


    plt.title(
        "BMI Distribution"
    )


    plt.tight_layout()


    plt.savefig(
        "static/bmi_distribution.png"
    )


    plt.close()


    # =====================================================
    # 3. SMOKER VS CHARGES
    # =====================================================

    smoker_charges = df.groupby(
        "smoker"
    )["charges"].mean()


    plt.figure(
        figsize=(8, 5)
    )


    smoker_charges.plot(
        kind="bar"
    )


    plt.xlabel(
        "Smoker"
    )


    plt.ylabel(
        "Average Insurance Charges"
    )


    plt.title(
        "Smoker vs Average Insurance Charges"
    )


    plt.tight_layout()


    plt.savefig(
        "static/smoker_charges.png"
    )


    plt.close()


    # =====================================================
    # 4. REGION VS CHARGES
    # =====================================================

    region_charges = df.groupby(
        "region"
    )["charges"].mean()


    plt.figure(
        figsize=(8, 5)
    )


    region_charges.plot(
        kind="bar"
    )


    plt.xlabel(
        "Region"
    )


    plt.ylabel(
        "Average Insurance Charges"
    )


    plt.title(
        "Region vs Average Insurance Charges"
    )


    plt.tight_layout()


    plt.savefig(
        "static/region_charges.png"
    )


    plt.close()


    # =====================================================
    # 5. BMI VS CHARGES
    # =====================================================

    plt.figure(
        figsize=(8, 5)
    )


    plt.scatter(
        df["bmi"],
        df["charges"]
    )


    plt.xlabel(
        "BMI"
    )


    plt.ylabel(
        "Insurance Charges"
    )


    plt.title(
        "BMI vs Insurance Charges"
    )


    plt.tight_layout()


    plt.savefig(
        "static/bmi_charges.png"
    )


    plt.close()


    # =====================================================
    # ANALYSIS HTML
    # =====================================================

    return render_template(
        "analysis.html"
    )


# =========================================================
# MODEL COMPARISON PAGE
# =========================================================

@app.route("/models")
def models():


    # -----------------------------------------------------
    # Load Model Comparison File
    # -----------------------------------------------------

    results = pd.read_csv(
        "model_comparison.csv"
    )


    # -----------------------------------------------------
    # Best Model
    # -----------------------------------------------------

    best_model = results.iloc[0]["Model"]


    # -----------------------------------------------------
    # Send Results to HTML
    # -----------------------------------------------------

    return render_template(

        "models.html",

        results=results.to_dict(
            orient="records"
        ),

        best_model=best_model
    )


# =========================================================
# ABOUT PAGE
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )