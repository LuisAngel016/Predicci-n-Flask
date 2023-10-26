import numpy as np
import joblib
from flask import Flask, render_template, request


def create_app():

    app = Flask(__name__)

    # Cargar el modelo
    with open('modelo_arbol_decision.pkl', 'rb') as archivo:
        arbol = joblib.load(archivo)

    @app.route('/')
    def index():
        return render_template('index.html')
        
    @app.route('/predict', methods=['POST'])   
    def predict():
        if request.method == 'POST':        
            try:
                # Capturar la entrada
                edad = int(request.form['edad'])
                genero = int(request.form['genero'])
                estrato = int(request.form['estrato'])
                materias_perdidas = int(request.form['materias_perdidas'])
                promedio = float(request.form['promedio'])
                # Crear un array con los datos de entrada        
                pred_args = [edad, genero, estrato, materias_perdidas, promedio]
                pred_arr = np.array(pred_args)
                preds = pred_arr.reshape(1, -1)
                # Realizar la predicción con el modelo cargado
                model_prediction = arbol.predict(preds)
                # Mostrar el resultado
                if model_prediction == 1:
                    res = "El estudiante desertará."
                else:
                    res = "El estudiante no desertará."
            except ValueError:
                return "Por favor, ingresa valores válidos"
            return render_template('prediccion.html', prediccion=res)
    
    return app

if __name__ == '__main__':
     app = create_app
     app.run()
