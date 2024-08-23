from flask import Flask, request, jsonify
from ingest import run_ingest
from chain import query

app = Flask(__name__)


@app.route('/query', methods=['POST'])
def route_query():
    data = request.get_json()
    response = query(data.get('query'))

    if response:
        return jsonify({"message": response}), 200

    return jsonify({"error": "Something went wrong"}), 400


if __name__ == '__main__':
    run_ingest()
    app.run(host="0.0.0.0", port=8080, debug=True)
    