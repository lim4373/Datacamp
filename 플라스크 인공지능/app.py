from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# 저장된 모델 불러오기
model = joblib.load('./knn_model.pkl')

@app.route('/')
def index():
    return render_template('index.html', predictions=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # HTML 폼에서 업로드한 dat 파일을 받습니다.
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # 업로드된 dat 파일을 읽어옵니다.
            uploaded_data = uploaded_file.read()
            uploaded_data = np.fromstring(uploaded_data, dtype=float, sep=' ')
            
            # 업로드된 데이터를 모델의 입력 형식에 맞게 변환합니다.
            if len(uploaded_data) == 4:  # 4개의 특성이 있는 경우
                # 각 클래스에 대한 확률 값을 얻습니다.
                probabilities = model.predict_proba([uploaded_data])
                class_probabilities = {f"Iris Class {i}": f"{prob * 100:.2f}%" for i, prob in enumerate(probabilities[0])}
                return render_template('index.html', predictions=class_probabilities)
            else:
                return render_template('index.html', predictions="Invalid data format")
        
        return render_template('index.html', predictions=None)
    except Exception as e:
        return render_template('index.html', predictions=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
