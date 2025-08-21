from flask import Flask, render_template, request
from views import views_bp

app = Flask(__name__)
app.register_blueprint(views_bp)  # Регистрируем Blueprint маршрутизатор

if __name__ == '__main__':
    app.run(debug=True)