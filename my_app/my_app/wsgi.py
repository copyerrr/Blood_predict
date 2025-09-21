from waitress import serve
from blood import app  # 여기서 'blood'는 Flask 애플리케이션이 정의된 모듈

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)
