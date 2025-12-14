from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/notification", methods=["POST"])
def notification():
    data = request.get_json()
    app.logger.info("Received notification: %s", data)
    # Here: route into your focus buddy system (e.g. enqueue, filter, show UI, persist)
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(port=5005)
