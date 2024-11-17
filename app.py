# app.py
from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('model.pkl')

# Initialize SHAP explainer
explainer = shap.Explainer(model)

# Define mappings
race_mapping = {
    'Non-Hispanic White': 0,
    'Non-Hispanic Black': 1,
    'Mexican American': 2,
    'Other Hispanic': 3,
    'Other Race': 4
}
education_mapping = {
    'Less than High School': 0,
    'High school': 1,
    'College and above': 2
}

app_ui = ui.page_fluid(
    ui.h2("Endometriosis Prediction Model"),

    ui.row(
        ui.column(6, ui.input_numeric("age", "Age", value=30, min=0, max=100)),
        ui.column(6, ui.input_select("race", "Race", choices=list(race_mapping.keys())))
    ),
    ui.row(
        ui.column(6, ui.input_numeric("mono", "Monocytes (Mono)", value=0.5)),
        ui.column(6, ui.input_numeric("plt_value", "Platelets (PLT)", value=200.0))
    ),
    ui.row(
        ui.column(6, ui.input_numeric("bmi", "BMI", value=25.0)),
        ui.column(6, ui.input_numeric("par", "PAR", value=0.1))
    ),
    ui.row(
        ui.column(6, ui.input_numeric("npar", "NPAR", value=0.1)),
        ui.column(6, ui.input_numeric("lmr", "LMR", value=2.0))
    ),
    ui.row(
        ui.column(6, ui.input_select("education", "Education Level", choices=list(education_mapping.keys())))
    ),
    ui.input_action_button("predict", "Predict"),
    ui.br(),
    ui.output_text_verbatim("prediction_output"),
    ui.output_plot("shap_force_plot")
)

def server(input, output, session):
    def get_input_data():
        # Map inputs
        race_num = race_mapping[input.race()]
        education_num = education_mapping[input.education()]

        # Create input data DataFrame
        input_data = pd.DataFrame({
            'Age': [input.age()],
            'Race': [race_num],
            'Mono': [input.mono()],
            'PLT': [input.plt_value()],
            'BMI': [input.bmi()],
            'PAR': [input.par()],
            'NPAR': [input.npar()],
            'LMR': [input.lmr()],
            'Education': [education_num]
        })

        return input_data

    @output
    @render.text
    @reactive.event(input.predict)
    def prediction_output():
        input_data = get_input_data()

        # Prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Prediction result
        if prediction[0] == 1:
            pred_text = 'Prediction: Endometriosis'
        else:
            pred_text = 'Prediction: No Endometriosis'
        prob_text = f'Risk of Endometriosis: {prediction_proba[0][1]:.2f}'

        return f"{pred_text}\n{prob_text}"

    @output
    @render.plot
    @reactive.event(input.predict)
    def shap_force_plot():
        input_data = get_input_data()

        # Calculate SHAP values
        shap_values = explainer(input_data)

        # Create force plot
        shap.plots.force(shap_values[0], matplotlib=True, show=False)
        plt.tight_layout()
        # Ensure the plot is displayed
        plt.show()

app = App(app_ui, server)
