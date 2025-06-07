import os
import re
import json
import markdown # For rendering Markdown from LLMs
from flask import Flask, request, jsonify, render_template
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
from werkzeug.utils import secure_filename
import uuid
import base64
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- API Key Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

# Initialize Groq client (for Llama)
# Using 'llama-3.1-70b-versatile' as requested. If not available, use 'llama3-70b-8192'.
GROQ_MODEL = "llama-3.3-70b-versatile"
# If llama-3.1-70b-versatile is not publicly available or causes issues, fallback to:
# GROQ_MODEL = "llama3-70b-8192"

groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize Gemini client
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-1.5-flash-latest" # Updated to use the latest flash model

# --- AI Interaction Functions ---

def generate_code_with_llama(prompt: str) -> tuple[str, bool]:
    """
    Generates code using Groq's Llama model.
    Returns: (generated_content_in_markdown, contains_code_block)
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are Ego, an unparalleled AI coding assistant master in frontend, AI/ML, app development, and backend. Generate comprehensive, production-quality, and highly efficient code. Always format code using markdown code blocks with language annotations (e.g., ```python). Provide only the necessary context and clear code."
                },
                {
                    "role": "user",
                    "content": f"Generate code for the following request: {prompt}"
                }
            ],
            model=GROQ_MODEL,
            temperature=0.7, # A bit more deterministic for code
            max_tokens=4000 # Max output tokens
        )
        content = chat_completion.choices[0].message.content
        # Check if the content contains a markdown code block
        contains_code_block = bool(re.search(r'```(?:\w+)?\n(.*?)\n```', content, re.DOTALL))
        return content, contains_code_block
    except Exception as e:
        print(f"Error generating code with Llama: {e}")
        return f"Error: Unable to generate code. {e}", False

def process_with_gemini(prompt: str, current_code: str = None, task_type: str = "explanation", images=None) -> tuple[str, bool]:
    """
    Processes requests using Gemini for explanation, modification, language learning, flowchart generation, or image analysis.
    task_type: "explanation", "modification", "learning", "flowchart", "image_analysis"
    Returns: (generated_text_or_code, success_status)
    """
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    system_prompt = (
        "You are Ego, a versatile AI assistant. Your expertise covers explaining code, modifying existing code (if provided), "
        "teaching programming languages effectively, generating precise execution flowcharts using Mermaid syntax, "
        "and analyzing images to generate code or provide explanations. "
        "Always be clear, helpful, and concise. For code modifications, aim to provide *only* the modified code block "
        "unless critical context is required. For explanations, be detailed but focused. For flowcharts, generate valid Mermaid syntax. "
        "For image analysis, describe what you see and generate appropriate code if requested."
    )

    # Create content parts in the format Gemini expects
    full_prompt = []
    
    # Add user content based on task type
    user_content = ""
    
    # Prepend system prompt to user content
    user_content = f"{system_prompt}\n\n"
    
    if task_type == "explanation":
        if current_code: # User provided code in the second textbox
            user_content += f"Explain the following code snippet and answer any specific questions: '{prompt}'.\n\nCode to explain:\n```\n{current_code}\n```\n\nPlease provide a detailed explanation of how this code works, including:\n1. The purpose of each major section\n2. How different parts interact\n3. Any important programming concepts used\n4. Potential improvements or best practices that could be applied"
        else:
            user_content += f"Please explain: {prompt}"
    
    elif task_type == "modification":
        if current_code:
            # Emphasize returning *only* the modified code block
            user_content += f"Given the following code block, please modify it according to this request: '{prompt}'. Provide only the complete modified code block, with proper markdown syntax (```language\ncode\n```), and no extra conversational text unless the modification is impossible or requires crucial caveats.\n\nCode to modify:\n```\n{current_code}\n```"
        else:
            return "Error: Cannot modify code without existing code provided.", False
            
    elif task_type == "learning":
        user_content += f"Help me learn {prompt}. Start with fundamental concepts, explain syntax, and provide a simple, practical example to illustrate. Make it beginner-friendly."
        
    elif task_type == "flowchart":
        user_content += f"Generate a flowchart representing the execution steps or project logic for: '{prompt}'. Use Mermaid syntax for the flowchart definition. Ensure the Mermaid code is within a ````mermaid\n...\n```` code block."
    
    elif task_type == "image_analysis":
        if images:
            user_content += f"Analyze the following image(s) and respond to this request: '{prompt}'. If the request involves generating code based on the image, please provide complete, functional code with proper explanations."
        else:
            return "Error: Cannot analyze images without image data provided.", False
    
    elif task_type == "error_analysis":
        if current_code and images:
            user_content += f"Analyze the following error shown in the image(s) and fix the code. User request: '{prompt}'.\n\nCode with error:\n```\n{current_code}\n```\n\nPlease provide:\n1. An explanation of what's causing the error\n2. A complete fixed version of the code\n3. Any additional recommendations to prevent similar errors"
        elif images:
            user_content += f"Analyze the error shown in the image(s). User request: '{prompt}'.\n\nPlease provide:\n1. An explanation of what's causing the error\n2. How to fix the error\n3. Any additional recommendations to prevent similar errors"
        else:
            return "Error: Cannot analyze errors without image data or code provided.", False
    
    else: # Default or unhandled Gemini use cases
        user_content += f"Process the following: {prompt}"
    
    # Create the parts list for the prompt
    parts = []
    parts.append({"text": user_content})
    
    # Add images if provided
    if images and (task_type == "image_analysis" or task_type == "error_analysis"):
        for image in images:
            parts.append({"inline_data": {"mime_type": image["mime_type"], "data": image["data"]}})
    
    # Add user message with proper format
    full_prompt.append({"role": "user", "parts": parts})

    try:
        response = model.generate_content(full_prompt, stream=False)
        return response.text, True
    except Exception as e:
        print(f"Error processing with Gemini ({task_type}): {e}")
        return f"Error processing your request with Gemini. {e}", False


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # Check if the request contains form data (multipart/form-data for image uploads)
    if request.content_type and 'multipart/form-data' in request.content_type:
        # Handle form data with possible image uploads
        user_prompt = request.form.get('prompt', '').strip()
        code_textbox_content = request.form.get('code_textbox_content', '').strip()
        current_displayed_code = request.form.get('current_displayed_code', '').strip()
        is_followup_on_code = request.form.get('is_followup_on_code') == 'true'
        
        # Process uploaded images
        uploaded_images = []
        if 'images' in request.files:
            image_files = request.files.getlist('images')
            for img_file in image_files:
                if img_file and img_file.filename:
                    # Read image data
                    img_data = img_file.read()
                    # Convert to base64 for Gemini API
                    encoded_img = base64.b64encode(img_data).decode('utf-8')
                    # Get MIME type
                    mime_type = img_file.content_type or 'image/jpeg'  # Default to JPEG if not specified
                    
                    uploaded_images.append({
                        "mime_type": mime_type,
                        "data": encoded_img
                    })
    else:
        # Handle JSON data (no images)
        data = request.json
        user_prompt = data.get('prompt', '').strip()
        code_textbox_content = data.get('code_textbox_content', '').strip() # Content from the second textbox
        current_displayed_code = data.get('current_displayed_code', '').strip() # Code on the right panel
        is_followup_on_code = data.get('is_followup_on_code', False) # Flag if it's a follow-up related to displayed code
        uploaded_images = []
    
    response_data = {
        'message': '',          # General text message / explanation
        'type': 'text',         # 'text', 'code', 'error', 'flowchart'
        'layout': 'default',    # 'default' or 'two-panel'
        'language': 'plaintext',
        'is_html_preview': False,
        'code_raw': ''          # The raw code content to display/update
    }

    if not user_prompt and not code_textbox_content and not uploaded_images:
        response_data['message'] = "Please provide a prompt, code to analyze, or an image."
        response_data['type'] = 'error'
        return jsonify(response_data), 400

    # --- Intelligent Prompt Routing ---
    # 0. Image Analysis (if images are uploaded)
    if uploaded_images:
        # Check if it's an error analysis request
        error_keywords = ["error", "bug", "fix", "issue", "problem", "not working", "debug", "solve"]
        is_error_analysis = any(keyword in user_prompt.lower() for keyword in error_keywords)
        
        if is_error_analysis and code_textbox_content:
            # Error analysis with code
            response_text, success = process_with_gemini(user_prompt, code_textbox_content, task_type="error_analysis", images=uploaded_images)
            response_data['message'] = markdown.markdown(response_text) if success else response_text
            response_data['type'] = 'text'
            
            # Check if response contains code block for fixing the error
            if success:
                match = re.search(r'```(?P<lang>\w+)?\n(?P<code>.*?)\n```', response_text, re.DOTALL)
                if match:
                    response_data['code_raw'] = match.group('code').strip()
                    response_data['language'] = match.group('lang') or 'plaintext'
                    response_data['layout'] = 'two-panel'
                    response_data['type'] = 'code'
                    
                    # Check for HTML content for preview tab
                    is_html_lang = response_data['language'].lower() in ['html', 'javascript', 'css', 'markup']
                    response_data['is_html_preview'] = is_html_lang
        elif is_error_analysis:
            # Error analysis without code
            response_text, success = process_with_gemini(user_prompt, task_type="error_analysis", images=uploaded_images)
            response_data['message'] = markdown.markdown(response_text) if success else response_text
            response_data['type'] = 'text'
        else:
            # General image analysis
            response_text, success = process_with_gemini(user_prompt, task_type="image_analysis", images=uploaded_images)
            response_data['message'] = markdown.markdown(response_text) if success else response_text
            response_data['type'] = 'text'
            
            # Check if response contains code block (e.g., for UI implementation from screenshot)
            if success:
                match = re.search(r'```(?P<lang>\w+)?\n(?P<code>.*?)\n```', response_text, re.DOTALL)
                if match:
                    response_data['code_raw'] = match.group('code').strip()
                    response_data['language'] = match.group('lang') or 'plaintext'
                    response_data['layout'] = 'two-panel'
                    response_data['type'] = 'code'
                    
                    # Check for HTML content for preview tab
                    is_html_lang = response_data['language'].lower() in ['html', 'javascript', 'css', 'markup']
                    response_data['is_html_preview'] = is_html_lang
        
        return jsonify(response_data)

    # 1. Direct commands: Learn a language or generate flowchart
    if "teach me" in user_prompt.lower():
        language_to_learn = user_prompt.lower().replace("teach me", "").strip()
        if not language_to_learn: language_to_learn = "programming basics"
        response_text, success = process_with_gemini(language_to_learn, task_type="learning")
        response_data['message'] = markdown.markdown(response_text) if success else response_text
        response_data['type'] = 'text'
        return jsonify(response_data)

    if "generate flowchart for" in user_prompt.lower() or "flowchart of" in user_prompt.lower():
        project_desc = user_prompt.lower().replace("generate flowchart for", "").replace("flowchart of", "").strip()
        response_text, success = process_with_gemini(project_desc, task_type="flowchart")
        if success and "```mermaid" in response_text: # Detect if Mermaid syntax was returned
            # Extract just the Mermaid code if it's wrapped, and tell frontend it's a flowchart
            mermaid_match = re.search(r'```mermaid\n(.*?)\n```', response_text, re.DOTALL)
            if mermaid_match:
                response_data['message'] = f'<pre class="mermaid hidden-from-syntax-highlighter">\n{mermaid_match.group(1).strip()}\n</pre>'
                response_data['type'] = 'flowchart'
            else: # If it had mermaid tag but not block, still send it as text but suggest user how to prompt.
                 response_data['message'] = markdown.markdown(response_text)
                 response_data['type'] = 'text'
        else:
            response_data['message'] = markdown.markdown(response_text) # Fallback to markdown
            response_data['type'] = 'text'
        return jsonify(response_data)

    # 2. Code Explanation (if `code_textbox_content` is provided)
    if code_textbox_content:
        explanation, success = process_with_gemini(user_prompt, code_textbox_content, task_type="explanation")
        response_data['message'] = markdown.markdown(explanation) if success else explanation
        response_data['type'] = 'text'
        response_data['layout'] = 'default' # Explanation usually reverts to default layout
        return jsonify(response_data)

    # 3. Code Modification (if a follow-up request to current displayed code)
    if is_followup_on_code and current_displayed_code:
        modification_prompt_lower = user_prompt.lower()
        # Expanded list of modification keywords to catch more modification requests
        modification_keywords = ["change", "update", "modify", "refactor", "add", "include", "insert", 
                               "color", "style", "format", "improve", "enhance", "fix", "adjust", 
                               "transform", "convert", "make it", "turn it", "set", "apply"]
        
        if any(keyword in modification_prompt_lower for keyword in modification_keywords):
            
            modified_code_markdown, success = process_with_gemini(user_prompt, current_displayed_code, task_type="modification")
            
            if success and "error" not in modified_code_markdown.lower():
                # Attempt to extract code block from Gemini's response
                match = re.search(r'```(?P<lang>\w+)?\n(?P<code>.*?)\n```', modified_code_markdown, re.DOTALL)
                if match:
                    response_data['code_raw'] = match.group('code').strip()
                    response_data['language'] = match.group('lang') or 'plaintext'
                    # Check for HTML content for preview tab
                    is_html_lang = response_data['language'].lower() in ['html', 'javascript', 'css', 'markup']
                    response_data['is_html_preview'] = is_html_lang
                    response_data['layout'] = 'two-panel'
                    response_data['type'] = 'code'
                    
                    # Gemini might sometimes include a brief intro before the code, display it in left panel
                    if modified_code_markdown.startswith('```') == False: # if content starts with text, parse markdown
                        response_data['message'] = markdown.markdown(modified_code_markdown.split('```')[0].strip())
                    
                else: # Gemini didn't return a clean code block, treat as text
                    response_data['message'] = markdown.markdown(modified_code_markdown)
                    response_data['type'] = 'text'
            else:
                response_data['message'] = "Ego couldn't modify the code as requested. Please be more specific or try regenerating."
                response_data['type'] = 'error'
            return jsonify(response_data)

    # 4. Code Generation (primary use for Llama)
    # This is a heuristic, in a real app you might use an LLM classifier for intent.
    generate_keywords = ["generate", "write", "create", "make me", "code for", "implement", "build a", "how to"]
    is_code_generation_request = any(keyword in user_prompt.lower() for keyword in generate_keywords) \
                                or ("show me" in user_prompt.lower() and "example" in user_prompt.lower())

    if is_code_generation_request:
        code_markdown, is_code_generated = generate_code_with_llama(user_prompt)
        response_data['layout'] = 'two-panel' # Always switch to two-panel for code generation

        if is_code_generated:
            # Extract main code block and language for syntax highlighting
            match = re.search(r'```(?P<lang>\w+)?\n(?P<code>.*?)\n```', code_markdown, re.DOTALL)
            if match:
                response_data['code_raw'] = match.group('code').strip()
                response_data['language'] = match.group('lang') or 'plaintext'
                
                # Check for HTML/JS/CSS for the preview tab
                is_html_lang = response_data['language'].lower() in ['html', 'javascript', 'css', 'markup']
                response_data['is_html_preview'] = is_html_lang

                # Display the full markdown (could contain explanations before code)
                response_data['message'] = markdown.markdown(code_markdown)
                response_data['type'] = 'code'
            else: # If Llama output markdown but not a clear code block (unlikely if system prompt is followed)
                response_data['message'] = markdown.markdown(code_markdown)
                response_data['type'] = 'text' # Still treat as text, not pure code for highlighting
        else: # Llama couldn't generate clear code or had an error
            response_data['message'] = markdown.markdown(code_markdown)
            response_data['type'] = 'text'
        return jsonify(response_data)
        
    # 5. General Queries/Explanation (fallback to Gemini)
    # If it wasn't a specific command, code explanation, code modification, or code generation,
    # assume it's a general question or explanation needed, use Gemini.
    else:
        general_response, success = process_with_gemini(user_prompt, task_type="explanation") # General explanation for now
        response_data['message'] = markdown.markdown(general_response) if success else general_response
        response_data['type'] = 'text'
        response_data['layout'] = 'default' # Always return to default layout for general answers
    
    return jsonify(response_data)

@app.route('/execute', methods=['POST'])
def execute_command():
    """Execute a command and return its output"""
    data = request.json
    command = data.get('command', '').strip()
    
    if not command:
        return jsonify({
            'success': False,
            'output': 'No command provided',
            'error': 'Command is required'
        }), 400
    
    try:
        # Execute the command and capture output
        import subprocess
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30  # Timeout after 30 seconds
        )
        
        return jsonify({
            'success': True,
            'output': result.stdout,
            'error': result.stderr,
            'exit_code': result.returncode
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'output': '',
            'error': 'Command execution timed out after 30 seconds',
            'exit_code': 124  # Standard timeout exit code
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'output': '',
            'error': str(e),
            'exit_code': 1
        })

if __name__ == '__main__':
    # Flask runs in debug mode if app.py is run directly
    # Ensure this is False in production environments.
    app.run(debug=True)