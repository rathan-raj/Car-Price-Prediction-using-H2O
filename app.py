import h2o
import gradio as gr
import pandas as pd
import os
import functools

# Set up Java path for H2O
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["PATH"] += os.pathsep + os.path.join(os.environ["JAVA_HOME"], "bin")

@functools.lru_cache(maxsize=1)
def get_model():
    h2o.init()
    model_path = os.path.join(os.path.dirname(__file__), "GBM_grid_1_AutoML_1_20250807_144050_model_2")
    return h2o.load_model(model_path)

features = ['Car Company Names', 'Cars Names', 'Engines', 'CC/Battery Capacity', 'HorsePower',
            'Total Speed', 'Performance(0 - 100 )KM/H', 'Fuel Types', 'Seats', 'Torque']

def predict_price(company_code, model_code, engine_code,
                  cc_battery, horsepower, top_speed,
                  perf_0_100, fuel_type_code, seats, torque):
    try:
        model = get_model()
        input_dict = {
            'Car Company Names': [int(company_code)],
            'Cars Names': [int(model_code)],
            'Engines': [int(engine_code)],
            'CC/Battery Capacity': [float(cc_battery)],
            'HorsePower': [float(horsepower)],
            'Total Speed': [float(top_speed)],
            'Performance(0 - 100 )KM/H': [float(perf_0_100)],
            'Fuel Types': [int(fuel_type_code)],
            'Seats': [int(seats)],
            'Torque': [float(torque)]
        }
        input_df = pd.DataFrame(input_dict)
        h2o_input = h2o.H2OFrame(input_df)
        h2o_input['Engines'] = h2o_input['Engines'].asfactor()
        h2o_input['Fuel Types'] = h2o_input['Fuel Types'].asfactor()
        prediction = model.predict(h2o_input)
        price_pred = prediction.as_data_frame().iloc[0, 0]
        return f"Estimated Price (USD): ${price_pred:,.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

# CSS
custom_css = """
body {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    font-family: 'Arial', sans-serif;
    color: #ffffff;
    margin: 0;
    padding: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
}
.gradio-container {
    max-width: 900px;
    margin: 1rem;
    padding: 2rem;
    background: rgba(10, 20, 50, 0.9);
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
h1 {
    text-align: center;
    color: #00c4ff;
    font-size: 2.8rem;
    font-weight: 700;
    text-shadow: 0 0 10px rgba(0, 196, 255, 0.7);
    margin-bottom: 0.5rem;
}
h3 {
    text-align: center;
    color: #d3d3d3;
    font-size: 1.1rem;
    font-weight: 400;
    margin-bottom: 2rem;
}
.gr-slider {
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 0.5rem;
}
.gr-slider > label > span {
    color: #00c4ff;
    font-weight: 600;
    font-size: 1rem;
}
.gr-slider input[type=range] {
    accent-color: #00c4ff;
    height: 8px;
    border-radius: 5px;
}
.gr-button {
    background: linear-gradient(90deg, #00c4ff, #0088ff);
    color: #ffffff;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1.2rem;
    padding: 0.8rem 1.5rem;
    border: none;
    box-shadow: 0 4px 15px rgba(0, 123, 255, 0.4);
    transition: all 0.3s ease;
}
.gr-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 123, 255, 0.6);
}
#component-15 {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 600;
    color: #00ffcc;
    padding: 1rem;
}
.gr-row {
    gap: 1rem;
}
.gr-column {
    background-color: rgba(0, 0, 0, 0.2);
    border-radius: 10px;
    padding: 1rem;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1>🚗 Car Price Estimator</h1>")
    gr.Markdown("<h3>Input your vehicle's specifications to get an accurate price estimate.</h3>")
    
    with gr.Row(variant='panel'):
        with gr.Column(scale=1):
            company_code = gr.Slider(0, 15, step=1, label="Car Company (encoded)")
            model_code = gr.Slider(0, 1000, step=1, label="Car Model (encoded)")
            engine_code = gr.Slider(0, 20, step=1, label="Engine Type (encoded)")
            cc_battery = gr.Slider(0, 7000, step=10, label="CC/Battery Capacity (cc/kWh)")
            horsepower = gr.Slider(0, 1200, step=1, label="HorsePower (HP)")
        with gr.Column(scale=1):
            top_speed = gr.Slider(0, 350, step=1, label="Top Speed (km/h)")
            zero_to_100 = gr.Slider(0, 20, step=0.1, label="0-100 km/h (seconds)")
            fuel_type_code = gr.Slider(0, 5, step=1, label="Fuel Type (encoded)")
            seats = gr.Slider(1, 8, step=1, label="Number of Seats")
            torque = gr.Slider(0, 1500, step=1, label="Torque (Nm)")
            
    with gr.Row():
        predict_btn = gr.Button("Estimate Price", variant="primary")

    output = gr.Textbox(label="Estimated Price", interactive=False)

    predict_btn.click(
        fn=predict_price,
        inputs=[company_code, model_code, engine_code, cc_battery, horsepower,
                top_speed, zero_to_100, fuel_type_code, seats, torque],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
