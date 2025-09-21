from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import difflib
import re
import io
import base64
import os

app = Flask(__name__)

class BloodSugarPredictor:
    def __init__(self, csv_file):
        self.is_model_switched = False
        self.is_diabetic = False

        # CSV 파일에서 데이터 읽기
        self.gi_data = pd.read_csv(csv_file, usecols=[0, 2, 3], header=None, names=["food_name", "carbohydrate", "gi_value"])
        self.gi_data['food_name'] = self.gi_data['food_name'].astype(str)

        # 신경망 모델 정의
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        self.X_train = []
        self.y_train = []
        self.current_blood_sugar = 0

        # 모델이 저장된 파일이 있으면 로드
        if os.path.exists('model.h5'):
            self.model = tf.keras.models.load_model('model.h5')

    def get_food_data(self, food_name):
        result = self.gi_data[self.gi_data['food_name'].str.lower() == food_name.lower()]
        if not result.empty:
            gi_value = float(result['gi_value'].values[0])
            carbohydrate_str = result['carbohydrate'].values[0]

            # 'g'와 같은 단위를 제거하고 숫자만 변환
            carbohydrate_str = re.sub(r'[^\d.]+', '', carbohydrate_str)  # 숫자와 점(.)만 남기기

            # 탄수화물 데이터가 숫자로 변환되지 않을 경우를 대비해 예외 처리
            try:
                carbohydrate_per_100g = float(carbohydrate_str)
            except ValueError:
                carbohydrate_per_100g = 0.0  # 기본값 설정

            return gi_value, carbohydrate_per_100g
        else:
            return None

    def suggest_similar_food(self, food_name):
        food_names = self.gi_data['food_name'].tolist()
        closest_matches = difflib.get_close_matches(food_name, food_names, n=1, cutoff=0.6)
        if closest_matches:
            return closest_matches[0]
        return None

    def predict_blood_sugar_with_formula(self, gi_value, actual_carbohydrate_amount, carbohydrate_per_100g):
        # 100g 기준의 탄수화물 양을 사용하여 공식 예측
        if carbohydrate_per_100g > 0:
            carbohydrate_ratio = actual_carbohydrate_amount / 100.0
        else:
            carbohydrate_ratio = 0

        if self.is_diabetic:
            increase_per_g_carbohydrate = gi_value * 0.01 * 2
        else:
            increase_per_g_carbohydrate = gi_value * 0.01

        total_increase = increase_per_g_carbohydrate * carbohydrate_ratio * carbohydrate_per_100g
        return total_increase + self.current_blood_sugar

    def predict_blood_sugar_with_model(self, actual_carbohydrate_amount, predicted_blood_sugar_formula):
        # 신경망 모델에 입력할 데이터 준비
        input_data = np.array([[actual_carbohydrate_amount, predicted_blood_sugar_formula]])

        # 모델을 사용하여 혈당 상승량 예측
        predicted_increase = self.model.predict(input_data)[0, 0]

        # 현재 혈당과 예측된 혈당 상승량을 더한 값을 반환
        return self.current_blood_sugar + predicted_increase

    def update_model(self, actual_carbohydrate_amount, predicted_blood_sugar_formula, actual_blood_sugar, current_blood_sugar):
        # 실제 혈당 상승량 계산
        blood_sugar_increase = actual_blood_sugar - current_blood_sugar

        # 입력 데이터 및 타겟 데이터를 학습 리스트에 추가
        self.X_train.append([actual_carbohydrate_amount, predicted_blood_sugar_formula])
        self.y_train.append(blood_sugar_increase)

        # 리스트를 numpy 배열로 변환
        X = np.array(self.X_train)
        y = np.array(self.y_train)

        # 신경망 모델 학습 (100번의 epoch 동안 학습 진행, verbose=1으로 학습 과정 출력)
        self.model.fit(X, y, epochs=100, verbose=1)

        # 학습된 모델을 파일로 저장
        self.model.save('model.h5')

    def switch_to_neural_network(self):
        self.is_model_switched = True

    def reset_training_data(self):
        self.X_train = []
        self.y_train = []

    def set_diabetic_status(self, is_diabetic):
        self.is_diabetic = is_diabetic

    def set_current_blood_sugar(self, current_blood_sugar):
        self.current_blood_sugar = current_blood_sugar

    def plot_graph(self, current_blood_sugar, predicted_blood_sugar_formula_total, predicted_blood_sugar_model_total):
        time_points = [0, 0.5, 1, 1.5, 2, 2.5, 3]

        # 신경망 모델을 이용한 혈당 예측, 최대값을 피크로 설정
        model_predictions = [
            current_blood_sugar,
            current_blood_sugar + (predicted_blood_sugar_model_total - current_blood_sugar) * 0.5,
            current_blood_sugar + (predicted_blood_sugar_model_total - current_blood_sugar) * 0.75,
            predicted_blood_sugar_model_total,  # 피크
            current_blood_sugar + (predicted_blood_sugar_model_total - current_blood_sugar) * 0.85,
            current_blood_sugar + (predicted_blood_sugar_model_total - current_blood_sugar) * 0.5,
            current_blood_sugar
        ]

        # 공식을 이용한 혈당 예측
        formula_predictions = [
            current_blood_sugar,
            current_blood_sugar + (predicted_blood_sugar_model_total - current_blood_sugar) * 0.8,
            predicted_blood_sugar_model_total,
            current_blood_sugar + (predicted_blood_sugar_model_total - current_blood_sugar) * 0.85,
            current_blood_sugar + (predicted_blood_sugar_model_total - current_blood_sugar) * 0.50,
            current_blood_sugar + (predicted_blood_sugar_model_total - current_blood_sugar) * 0.20,
            current_blood_sugar
        ]

        plt.figure(figsize=(6, 2))

        # 당뇨 여부에 따른 그래프 표시
        if self.is_diabetic:
            plt.plot(time_points, model_predictions, 'purple', label='당뇨환자 (신경망 예측)', marker='o')
        else:
            plt.plot(time_points, formula_predictions, 'red', label='일반인 (공식 예측)', marker='o')

        plt.title('식후 혈당 예측')
        plt.xlabel('시간 (시간)')
        plt.ylabel('혈당 (mg/dL)')
        plt.xticks(time_points)
        plt.legend()
        plt.grid(True)

        # Plot을 이미지로 저장
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # 이미지 데이터를 base64로 인코딩
        plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
        return plot_url

    def repeated_training(self, actual_carbohydrate_amount, predicted_blood_sugar_formula, actual_blood_sugar, current_blood_sugar):
        while True:
            # 신경망 모델을 사용하여 혈당 예측
            predicted_blood_sugar_model_total = self.predict_blood_sugar_with_model(actual_carbohydrate_amount, predicted_blood_sugar_formula)

            # 신경망 모델의 예측과 실제 혈당의 오차 계산
            error = abs(predicted_blood_sugar_model_total - actual_blood_sugar)

            if error <= 0.5:
                print(f"예상 혈당 (신경망 모델 사용): {predicted_blood_sugar_model_total:.3f} mg/dL")
                self.switch_to_neural_network()
                self.reset_training_data()
                break
            else:
                print(f"신경망 모델 예측된 혈당({predicted_blood_sugar_model_total:.3f} mg/dL)이 현재 혈당({actual_blood_sugar:.3f} mg/dL)과의 오차가 {error:.3f} mg/dL입니다. 모델을 추가 학습합니다.")
                self.update_model(actual_carbohydrate_amount, predicted_blood_sugar_formula, actual_blood_sugar, current_blood_sugar)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        food_name = request.form['food_name']
        carbohydrate_amount = float(request.form['carbohydrate_amount'])
        current_blood_sugar = float(request.form['current_blood_sugar'])
        diabetic_status = request.form['diabetic_status']

        is_diabetic = diabetic_status == 'yes'

        # BloodSugarPredictor 인스턴스 생성 및 설정
        predictor = BloodSugarPredictor('BloodSugarPredictor.csv')
        predictor.set_current_blood_sugar(current_blood_sugar)
        predictor.set_diabetic_status(is_diabetic)

        # 음식 데이터 가져오기
        food_data = predictor.get_food_data(food_name)

        if food_data:
            gi_value, carbohydrate_per_100g = food_data
            predicted_blood_sugar_formula_total = predictor.predict_blood_sugar_with_formula(gi_value, carbohydrate_amount, carbohydrate_per_100g)
            predicted_blood_sugar_model_total = predictor.predict_blood_sugar_with_model(carbohydrate_amount, predicted_blood_sugar_formula_total)

            # 그래프 생성
            plot_url = predictor.plot_graph(current_blood_sugar, predicted_blood_sugar_formula_total, predicted_blood_sugar_model_total)

            # 예측 결과 페이지로 리디렉션
            return render_template('index.html',
                                   prediction_ready=True,
                                   predicted_blood_sugar_formula_total=predicted_blood_sugar_formula_total,
                                   predicted_blood_sugar_model_total=predicted_blood_sugar_model_total,
                                   plot_url=plot_url,
                                   food_name=food_name,
                                   carbohydrate_amount=carbohydrate_amount,
                                   current_blood_sugar=current_blood_sugar)
        else:
            similar_food = predictor.suggest_similar_food(food_name)
            return render_template('index.html', error="음식을 찾을 수 없습니다.", similar_food=similar_food)
    except KeyError as e:
        return f"Missing form data: {e}", 400
    except ValueError as e:
        return f"Invalid input: {e}", 400

@app.route('/update_model', methods=['POST'])
def update_model():
    try:
        food_name = request.form['food_name']
        carbohydrate_amount = float(request.form['carbohydrate_amount'])
        current_blood_sugar = float(request.form['current_blood_sugar'])
        actual_blood_sugar = float(request.form['actual_blood_sugar'])

        # BloodSugarPredictor 인스턴스 생성 및 설정
        predictor = BloodSugarPredictor('BloodSugarPredictor.csv')
        predictor.set_current_blood_sugar(current_blood_sugar)

        # 음식 데이터 가져오기
        food_data = predictor.get_food_data(food_name)

        if food_data:
            gi_value, carbohydrate_per_100g = food_data
            predicted_blood_sugar_formula_total = predictor.predict_blood_sugar_with_formula(gi_value, carbohydrate_amount, carbohydrate_per_100g)

            # 모델 업데이트
            predictor.repeated_training(carbohydrate_amount, predicted_blood_sugar_formula_total, actual_blood_sugar, current_blood_sugar)

            return render_template('index.html',
                                   prediction_ready=False,  # Reset for a new prediction
                                   success="모델이 성공적으로 학습되었습니다.",
                                   food_name=food_name,
                                   carbohydrate_amount=carbohydrate_amount,
                                   current_blood_sugar=current_blood_sugar)
        else:
            return render_template('index.html', error="음식을 찾을 수 없습니다.")
    except KeyError as e:
        return f"Missing form data: {e}", 400
    except ValueError as e:
        return f"Invalid input: {e}", 400



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)