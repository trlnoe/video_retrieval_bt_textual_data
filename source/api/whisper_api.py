import flask
from flask import jsonify, request
#libraries 
import whisper

#funtions 
app = flask.Flask("API for ARI Youtube")
app.config["DEBUG"] = True
model = whisper.load_model("medium")

@app.route('/speech2text', methods=['POST', 'GET'])
def updateCurrentCode():
    global model
    audio_path = '/workspace/competitions/AI_Challenge_2022/source/util/test_whisper.mp3'
    if request.method == "POST":
        text = request.json['new_text']
    else:
        text = request.args.get('new_text')
    result = model.transcribe(audio_path)
    print('audio text: ', result["text"])  
    response = flask.jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  
#arguments

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8020, debug=False)