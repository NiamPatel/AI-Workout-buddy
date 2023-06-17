from flask import Flask, render_template, redirect, url_for, send_from_directory

app = Flask(__name__)

button_urls = {
    'button1': 'http://127.0.0.1:1000',
    'button2': 'http://127.0.0.1:2000',
    'button3': 'http://127.0.0.1:3000',
    'button4': 'http://127.0.0.1:4000',
    'button5': 'http://127.0.0.1:5000',
    'button6': 'http://127.0.0.1:6000',

}

button_states = {
    'Push Ups': False,
    'Sit Ups': False,
    'Squats': False,
    'Plank': False,
    'Lungs': False,
    'Bisep Curls': False,

}

Instructions = "To use this web app, simply press the button corresponding to your workout and position your camera so that it can see your whole body. After working out, check the website to get feedback on your form and ways to improve your workout. Created by Niam Patel"

@app.route('/')
def index():
    return render_template('index.html', button_states=Instructions, Instructions=Instructions)

@app.route('/button_pressed/<button_id>')
def button_pressed(button_id):
    button_states[button_id] = True
    return redirect(button_urls[button_id])

@app.route('/back.jpg')
def serve_background_image():
    return send_from_directory('static', 'back.jpg')
@app.route('/501260.png')
def serve_image_501260():
    return send_from_directory('static', '501260.png')
@app.route('/situp.png')
def serve_image_situp():
    return send_from_directory('static', 'situp.png')
@app.route('/squat.png')
def serve_image_squat():
    return send_from_directory('static', 'squat.png')
@app.route('/plank.png')
def serve_image_plank():
    return send_from_directory('static', 'plank.png')
@app.route('/b.png')
def serve_image_b():
    return send_from_directory('static', 'b.png')
@app.route('/lunge.jpg')
def serve_image_lunge():
    return send_from_directory('static', 'lunge.jpg')

if __name__ == '__main__':
    app.run(port=1111)
