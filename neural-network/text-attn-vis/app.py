from flask import Flask, request, render_template
import nltk
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
#run_with_ngrok(app)

@app.route('/')
def my_form():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="172.28.0.2", port=8000)
    #172.28.0.2 
    #python -m http.server
    #app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)