from flask import Flask, request, jsonify
import os
import tempfile
import logging
from analyzer import DataAnalyzer
from llm_handler import LLMHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Test route to verify server is working
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "API is running",
        "endpoints": {
            "POST /api/": "Main data analysis endpoint"
        }
    })

# Main API endpoint
@app.route('/api/', methods=['POST', 'GET'])
def analyze_data():
    if request.method == 'GET':
        return jsonify({
            "message": "Data Analyst Agent API",
            "usage": "Send POST request with questions.txt file",
            "example": "curl -X POST /api/ -F 'questions.txt=@questions.txt'"
        })
    
    try:
        logger.info("Received POST request to /api/")
        logger.info(f"Request files: {list(request.files.keys())}")
        logger.info(f"Request form: {dict(request.form)}")
        
        # Get files from request
        files = {}
        questions_text = ""
        
        # Extract questions.txt (always present)
        if 'questions.txt' in request.files:
            questions_file = request.files['questions.txt']
            questions_text = questions_file.read().decode('utf-8')
            logger.info(f"Questions text: {questions_text[:200]}...")
        else:
            logger.error("questions.txt not found in request")
            return jsonify({"error": "questions.txt is required"}), 400
        
        # Extract other files
        for filename in request.files:
            if filename != 'questions.txt':
                files[filename] = request.files[filename]
                logger.info(f"Additional file: {filename}")
        
        # Initialize analyzer and LLM handler
        try:
            analyzer = DataAnalyzer()
            llm = LLMHandler()
            logger.info("Initialized analyzer and LLM handler")
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return jsonify({"error": f"Initialization failed: {str(e)}"}), 500
        
        # Process the request
        logger.info("Starting data analysis...")
        result = analyzer.process_request(questions_text, files, llm)
        logger.info(f"Analysis complete. Result type: {type(result)}")
        
        # Ensure result is JSON serializable
        if isinstance(result, dict) and "error" in result:
            logger.error(f"Analysis error: {result['error']}")
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Unexpected error in analyze_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Additional test endpoints for debugging
@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({"message": "Test endpoint working"})

@app.route('/api/test', methods=['POST'])
def test_api():
    return jsonify({
        "message": "API test endpoint working",
        "files_received": list(request.files.keys()),
        "form_data": dict(request.form)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting server on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)