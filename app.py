from flask import Flask, render_template, request
import os
from utils import predict_image, get_disease_info

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filepath = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(filepath)

            prediction, confidence = predict_image(filepath)
            info = get_disease_info(prediction)

            return render_template('index.html',
                                   prediction=prediction,
                                   confidence=confidence,
                                   filename=image.filename,
                                   info=info)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)




#1. `from flask import Flask, render_template, request` → Import Flask and tools to render HTML and handle user requests.
#2. `import os` → Import OS module to manage files and directories.
#3. `from utils import predict_image, get_disease_info` → Import custom functions for predicting disease from an image and fetching info.
#4. `app = Flask(__name__)` → Create a Flask app instance to run the web application.
#5. `UPLOAD_FOLDER = 'static/uploads'` → Define folder where uploaded images will be stored.
#6. `os.makedirs(UPLOAD_FOLDER, exist_ok=True)` → Create the upload folder if it doesn’t exist yet.
#7. `@app.route('/', methods=['GET', 'POST'])` → Connect the URL `/` to this function; it handles:
#* **GET** → When user visits the page, show the form/page.
#* **POST** → When user submits the form, process the uploaded image.
#8. `def home():` → Function that contains the logic for displaying and processing the homepage.
  #9. `if request.method == 'POST':` → Check if the form has been submitted (user uploaded an image).
  #10. `image = request.files['image']` → Retrieve the uploaded image file from the form data.
  #11. `if image:` → Ensure that a file was actually uploaded before processing.
  #12. `filepath = os.path.join(UPLOAD_FOLDER, image.filename)` → Create a safe path to save the uploaded file.
  #13. `image.save(filepath)` → Save the uploaded file to the server in the upload folder.
  #14. `prediction, confidence = predict_image(filepath)` → Use the model to predict disease and confidence score.
  #15. `info = get_disease_info(prediction)` → Get extra information about the detected disease.
  #16. `return render_template('index.html', prediction=prediction, confidence=confidence, filename=image.filename, info=info)` → Render the homepage with the prediction results and image.
  #17. `return render_template('index.html')` → Render the homepage without results (initial load or no file uploaded).
#18. `if __name__ == '__main__':` → Ensure the app runs only if this script is executed directly.
#19. `app.run(debug=True)` → Start the Flask server with debug mode (auto-reload and error messages).

