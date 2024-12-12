from flask import Flask, jsonify
from papertest import getPostion
# Initialize the Flask app
app = Flask(__name__)


@app.route('/get_stock_data', methods=['GET'])
def get_positions():
    try:
        # Fetch positions using the getPosition function
        positions = getPostion()
        return jsonify({"status": "success", "data": positions}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
