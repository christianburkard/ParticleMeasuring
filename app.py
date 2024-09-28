from flask import Flask, render_template, redirect, url_for, flash, send_file, session, after_this_request
from flask_wtf import FlaskForm
from wtforms import MultipleFileField, SubmitField, SelectField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired, DataRequired
from wtforms import RadioField
from main import capsSegMain  # Import the function from main.py
from pathlib import Path
import zipfile
import shutil
import uuid
import os
import pandas as pd
from functions import plot_data
from app_tasks import get_pix_to_um, merge_data
from rq import Queue
from redis import Redis
from processImage import process_images
import pdb
from celery import Celery, Task
from config import make_celery

# Define allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/files'  # Set upload folder path

# Initialize Flask app and queue
app = Flask(__name__)

app.config['SECRET_KEY'] = 'supersecretkey'  # Secret key for session management and CSRF protection
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Configure upload folder
app.config['CELERY_CONFIG'] = {'broker_url': 'redis://172.29.237.14:6379', 
                               'result_backend':'redis://172.29.237.14:6379'}

celery = make_celery(app)
celery.set_default()
class homeForm(FlaskForm):
    files = MultipleFileField("", validators=[InputRequired()])
    magnification = SelectField("Magnification", choices=[('X20', 'X20'), ('X30', 'X30'), ('X40', 'X40'), ('X50', 'X50'), ('X80', 'X80'), ('X100', 'X100')], validators=[DataRequired()])
    resolution = RadioField("Resolution", choices=[('normal', 'Normal'), ('hd', 'HD')], default='normal', validators=[DataRequired()])
    submit = SubmitField("Analyze Images")

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Assume you have a user loader and password verification function
def verify_login(username, password):
    # Replace with actual user verification logic
    return username == 'admin' and password == 'password'

# Modify the home route to call the new function
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = homeForm()
    if form.validate_on_submit():
        files = form.files.data
        magnification = form.magnification.data
        resolution = form.resolution.data  # Get the selected resolution (either "normal" or "hd")

        pix_to_um = get_pix_to_um(magnification, resolution)

        # The rest of the code remains unchanged
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        filepaths = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(session_folder, filename)
                file.save(filepath)
                filepaths.append(filepath)

        if filepaths:
            # Call the new process_images function
            pdb.set_trace()
            image_pairs, summary_plot_path, summary_df = process_images.delay(filepaths, pix_to_um, session_folder)

            if image_pairs:
                return render_template('result.html', image_pairs=image_pairs, session_id=session_id, summary_plot=f'{session_id}/summary_plot.png', summary_df=summary_df)
            else:
                flash("No processed images found.")
                shutil.rmtree(session_folder)
                return redirect(url_for('home'))
        else:
            flash("Invalid file extension(s).")
            shutil.rmtree(session_folder)
            return redirect(url_for('home'))
    return render_template('index.html', form=form)



@app.route('/download/<session_id>')
def download(session_id):
    session_folder = Path(app.config['UPLOAD_FOLDER']) / session_id
    zip_path = session_folder / 'all_files.zip'
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in session_folder.glob('*'):
            if file.name != 'all_files.zip':
                zipf.write(file, arcname=file.name)
    
    @after_this_request
    def remove_file(response):
        try:
            shutil.rmtree(session_folder)  # Clean up by removing the session folder after download
        except Exception as e:
            app.logger.error("Error removing or closing downloaded file handle", e)
        return response

    return send_file(zip_path, as_attachment=True)

@app.route('/cleanup')
def cleanup():
    session_id = session.get('session_id')
    if session_id:
        session_folder = Path(app.config['UPLOAD_FOLDER']) / session_id
        if session_folder.exists():
            shutil.rmtree(session_folder)
    return redirect(url_for('home'))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

   
