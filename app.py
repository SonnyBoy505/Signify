#importamos la libreria de flask 
from flask import Flask ,render_template, Response
from main import main
#Creacion de la aplicacion
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route("/login")
def login():
    return render_template("login.html")

@app.route('/video')
def video():
    #Mandamos a llamar la funcion que muestra el video
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=image')

@app.route('/QuienesSomos')
def QuienesSomos():
    return render_template("quienesSomos.html")

@app.route('/f')
def pagina():
    return render_template('f.html')

#Ejecutar la app
if __name__ == "__main__":
    app.run(debug=True)
