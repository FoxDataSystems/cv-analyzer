from openai import AzureOpenAI
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_file, make_response
import os
import sqlite3
import tempfile
import PyPDF2
import docx
import uuid
from werkzeug.utils import secure_filename
import json
from datetime import datetime
from jinja2 import Template
import pdfkit
# Import the generate_cv function from cv_generator.py
from cv_generator import generate_cv

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key="xxx",  
    api_version="xxx",
    azure_endpoint="xxx"
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database setup
def init_db():
    conn = sqlite3.connect('cv_database.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS candidates (
        id TEXT PRIMARY KEY,
        name TEXT,
        email TEXT,
        phone TEXT,
        cv_text TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS work_experience (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id TEXT,
        company TEXT,
        position TEXT,
        start_date TEXT,
        end_date TEXT,
        description TEXT,
        FOREIGN KEY (candidate_id) REFERENCES candidates (id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS skills (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id TEXT,
        skill_name TEXT,
        years_experience REAL,
        FOREIGN KEY (candidate_id) REFERENCES candidates (id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS certificates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id TEXT,
        name TEXT,
        issuer TEXT,
        date_obtained TEXT,
        expiry_date TEXT,
        description TEXT,
        FOREIGN KEY (candidate_id) REFERENCES candidates (id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    """Extract text from PDF, DOCX, or TXT files"""
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    elif file_extension in ['docx', 'doc']:
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    elif file_extension == 'txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    return ""

def get_cv_analysis_prompt(cv_text):
    return f"""Please analyze this CV text and extract the following information in JSON format. For each certificate, analyze what technologies or skills it represents and add those to the skills section.

    {{
        "name": "candidate's full name",
        "email": "email address",
        "phone": "phone number",
        "work_experience": [
            {{
                "company": "company name",
                "position": "job title",
                "start_date": "start date",
                "end_date": "end date or 'Present'",
                "description": "job description"
            }}
        ],
        "skills": [
            {{
                "skill_name": "name of skill",
                "years_experience": "number of years (decimal)"
            }}
        ],
        "certificates": [
            {{
                "name": "name of certification/training",
                "issuer": "issuing organization/institution",
                "date_obtained": "date obtained (if available)",
                "expiry_date": "expiration date (if applicable)",
                "description": "brief description or additional details",
                "related_skills": ["list of skills this certificate represents"]
            }}
        ]
    }}

    Please ensure:
    1. All dates are in YYYY-MM format when possible
    2. Years of experience are numbers (can be decimals)
    3. Include all relevant certifications, professional qualifications, and training programs
    4. If certain fields are not available, use empty strings or null
    5. For ongoing certifications, use 'Present' for expiry_date
    6. Analyze each certification and add corresponding skills to the skills section. For example:
       - DP-600 certification should add "Microsoft Fabric" and "Data Analytics" to skills
       - AWS Solutions Architect should add "AWS", "Cloud Architecture" to skills
       - CISSP should add "Information Security", "Cybersecurity" to skills
    7. When adding certificate-based skills, set the years_experience to match the time since certification obtained
    8. Ensure no duplicate skills are added (merge and take the highest years of experience if found in multiple sources)

    Example of certificate analysis:
    If someone has "Microsoft Certified: Azure Data Engineer Associate (DP-203)" from 2021-06:
    - Add to certificates array with all details
    - Add to skills: "Azure Data Factory", "Azure Synapse", "Data Engineering", "Azure" (with ~2 years experience based on certification date)

    CV Text:
    {cv_text}
    """

def process_cv_response(gpt_response):
    """Process the GPT response and merge any duplicate skills"""
    data = json.loads(gpt_response)
    
    # Create a dictionary to track skills and their maximum years of experience
    skills_dict = {}
    
    # Process existing skills
    for skill in data.get('skills', []):
        skill_name = skill['skill_name'].lower()
        years = float(skill.get('years_experience', 0))
        skills_dict[skill_name] = max(skills_dict.get(skill_name, 0), years)
    
    # Process skills from certificates
    for cert in data.get('certificates', []):
        cert_date = cert.get('date_obtained', '')
        if cert_date:
            try:
                # Calculate years of experience based on certification date
                cert_date = datetime.strptime(cert_date, '%Y-%m')
                years_since_cert = (datetime.now() - cert_date).days / 365.25
                
                # Add related skills with years since certification
                for skill in cert.get('related_skills', []):
                    skill_name = skill.lower()
                    skills_dict[skill_name] = max(skills_dict.get(skill_name, 0), years_since_cert)
            except ValueError:
                # Handle invalid date format
                pass
    
    # Convert back to list format
    data['skills'] = [
        {'skill_name': skill, 'years_experience': round(years, 1)}
        for skill, years in skills_dict.items()
    ]
    
    return data

def analyze_cv_with_ai(cv_text):
    """Use Azure OpenAI to extract information from CV"""
    prompt = get_cv_analysis_prompt(cv_text)
    
    response = client.chat.completions.create(
        model="gpt-4",  # Use the appropriate model available in your Azure OpenAI deployments
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts structured information from CVs. Use Dutch terms only. Try to extract as many skills as possible. Voeg Standaard skills toe als deze niet in de CV staan. Zoals Analytische vaardigheden, communicatievaardigheden, Nederlands, Engels, zelfstandig werken, etc. Kunstmatige Intelligentie is AI/Artificial Intelligence."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    return response.choices[0].message.content

def save_to_database(cv_data):
    """Save the extracted CV data to the database"""
    conn = sqlite3.connect('cv_database.db')
    cursor = conn.cursor()
    
    candidate_id = str(uuid.uuid4())
    
    # Insert candidate information
    cursor.execute(
        "INSERT INTO candidates (id, name, email, phone, cv_text) VALUES (?, ?, ?, ?, ?)",
        (candidate_id, cv_data.get('name', ''), cv_data.get('email', ''), 
         cv_data.get('phone', ''), cv_data.get('cv_text', ''))
    )
    
    # Insert work experience
    for exp in cv_data.get('work_experience', []):
        cursor.execute(
            "INSERT INTO work_experience (candidate_id, company, position, start_date, end_date, description) VALUES (?, ?, ?, ?, ?, ?)",
            (candidate_id, exp.get('company', ''), exp.get('position', ''), 
             exp.get('start_date', ''), exp.get('end_date', ''), exp.get('description', ''))
        )
    
    # Insert skills
    for skill in cv_data.get('skills', []):
        cursor.execute(
            "INSERT INTO skills (candidate_id, skill_name, years_experience) VALUES (?, ?, ?)",
            (candidate_id, skill.get('skill_name', ''), skill.get('years_experience', 0))
        )
    
    # Insert certificates/trainings
    for cert in cv_data.get('certificates', []):
        cursor.execute(
            "INSERT INTO certificates (candidate_id, name, issuer, date_obtained, expiry_date, description) VALUES (?, ?, ?, ?, ?, ?)",
            (candidate_id, cert.get('name', ''), cert.get('issuer', ''), 
             cert.get('date_obtained', ''), cert.get('expiry_date', ''), cert.get('description', ''))
        )
    
    conn.commit()
    conn.close()
    
    return candidate_id

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('cv_database.db')
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()

    # Get total candidates
    cursor.execute("SELECT COUNT(*) FROM candidates")
    total_candidates = cursor.fetchone()[0]

    # Get unique skills count
    cursor.execute("SELECT COUNT(DISTINCT skill_name) FROM skills")
    unique_skills = cursor.fetchone()[0]
    
    # Get most experienced skill (highest average years)
    cursor.execute("""
        SELECT skill_name, AVG(years_experience) as avg_exp
        FROM skills
        GROUP BY skill_name
        HAVING COUNT(*) > 2  -- Only consider skills that appear multiple times
        ORDER BY avg_exp DESC
        LIMIT 1
    """)
    most_exp_skill = cursor.fetchone()
    most_experienced_skill = {
        'name': most_exp_skill['skill_name'] if most_exp_skill else 'None',
        'years': round(most_exp_skill['avg_exp'], 1) if most_exp_skill else 0
    }
    
    # Get candidate with most skills
    cursor.execute("""
        SELECT c.name, COUNT(s.id) as skill_count
        FROM candidates c
        JOIN skills s ON c.id = s.candidate_id
        GROUP BY c.id
        ORDER BY skill_count DESC
        LIMIT 1
    """)
    most_skilled = cursor.fetchone()
    most_skilled_candidate = {
        'name': most_skilled['name'] if most_skilled else 'None',
        'count': most_skilled['skill_count'] if most_skilled else 0
    }

    # Get top skills
    cursor.execute("""
        SELECT skill_name, COUNT(*) as count
        FROM skills
        GROUP BY skill_name
        ORDER BY count DESC
        LIMIT 6
    """)
    skills_data = cursor.fetchall()
    
    # Calculate percentages for skills
    max_count = max([skill['count'] for skill in skills_data]) if skills_data else 1
    top_skills = [
        {
            'name': skill['skill_name'],
            'count': skill['count'],
            'percentage': (skill['count'] / max_count) * 100
        }
        for skill in skills_data
    ]
    
    # Get top candidates with their most frequent skill
    cursor.execute("""
        SELECT c.id, c.name, s.skill_name, COUNT(s.skill_name) as skill_count
        FROM candidates c
        JOIN skills s ON c.id = s.candidate_id
        GROUP BY c.id, s.skill_name
        ORDER BY skill_count DESC
    """)
    all_candidate_skills = cursor.fetchall()
    
    # Process to get top skill for each candidate
    candidate_top_skills = {}
    for row in all_candidate_skills:
        if row['id'] not in candidate_top_skills:
            candidate_top_skills[row['id']] = {
                'name': row['name'],
                'top_skill': row['skill_name']
            }
    
    # Get top 15 candidates (or all if less than 15)
    top_candidates = list(candidate_top_skills.values())[:15]
    
    # Get recent activities
    recent_activities = []
    
    # Get most recent candidates
    cursor.execute("""
        SELECT id, name
        FROM candidates
        ORDER BY id DESC
        LIMIT 4
    """)
    recent_candidates = cursor.fetchall()
    
    # Create activity entries for recent candidates
    for candidate in recent_candidates:
        recent_activities.append({
            'icon': 'file-earmark-person',
            'title': 'Nieuw kandidaatprofiel aangemaakt',
            'description': f"{candidate['name']} is toegevoegd aan de database",
            'time': 'Recent'
        })

    conn.close()

    return render_template('dashboard.html',
                         total_candidates=total_candidates,
                         unique_skills=unique_skills,
                         most_experienced_skill=most_experienced_skill,
                         most_skilled_candidate=most_skilled_candidate,
                         top_skills=top_skills,
                         top_candidates=top_candidates,
                         recent_activities=recent_activities)

@app.route('/upload')
def upload_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    files = request.files.getlist('files[]')
    
    if not files or files[0].filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    processed_count = 0
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                # Extract text from the file
                cv_text = extract_text_from_file(file_path)
                
                # Analyze the CV with Azure OpenAI
                response = analyze_cv_with_ai(cv_text)
                processed_data = process_cv_response(response)
                processed_data['cv_text'] = cv_text  # Add the original text for reference
                
                # Save to database
                save_to_database(processed_data)
                
                processed_count += 1
            except Exception as e:
                flash(f'Error processing {filename}: {str(e)}')
            
            # Clean up the uploaded file
            os.remove(file_path)
    
    if processed_count > 0:
        flash(f'Successfully processed {processed_count} CV(s)')
    
    return redirect(url_for('view_candidates'))

@app.route('/candidates')
def view_candidates():
    conn = sqlite3.connect('cv_database.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, email, phone FROM candidates")
    candidates = cursor.fetchall()
    
    conn.close()
    
    return render_template('candidates.html', candidates=candidates)

@app.route('/candidate/<candidate_id>')
def view_candidate(candidate_id):
    conn = sqlite3.connect('cv_database.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get candidate info
    cursor.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,))
    candidate = cursor.fetchone()
    
    # Get work experience
    cursor.execute("SELECT * FROM work_experience WHERE candidate_id = ?", (candidate_id,))
    work_experience = cursor.fetchall()
    
    # Get skills
    cursor.execute("SELECT * FROM skills WHERE candidate_id = ?", (candidate_id,))
    skills = cursor.fetchall()
    
    # Get certificates
    cursor.execute("SELECT * FROM certificates WHERE candidate_id = ? ORDER BY date_obtained DESC", (candidate_id,))
    certificates = cursor.fetchall()
    
    conn.close()
    
    return render_template('candidate_details.html', 
                          candidate=candidate, 
                          work_experience=work_experience, 
                          skills=skills,
                          certificates=certificates)

@app.route('/candidate/delete/<candidate_id>', methods=['POST'])
def delete_candidate(candidate_id):
    conn = sqlite3.connect('cv_database.db')
    cursor = conn.cursor()
    
    # Delete skills
    cursor.execute("DELETE FROM skills WHERE candidate_id = ?", (candidate_id,))
    
    # Delete work experience
    cursor.execute("DELETE FROM work_experience WHERE candidate_id = ?", (candidate_id,))
    
    # Delete candidate
    cursor.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
    
    conn.commit()
    conn.close()
    
    flash('Candidate deleted successfully')
    return redirect(url_for('view_candidates'))

@app.route('/edit_candidate/<candidate_id>', methods=['GET', 'POST'])
def edit_candidate(candidate_id):
    if request.method == 'POST':
        # Update candidate information
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        
        conn = sqlite3.connect('cv_database.db')
        cursor = conn.cursor()
        
        # Update candidate
        cursor.execute(
            "UPDATE candidates SET name = ?, email = ?, phone = ? WHERE id = ?",
            (name, email, phone, candidate_id)
        )
        
        # Handle work experience updates
        # First, delete all existing work experience
        cursor.execute("DELETE FROM work_experience WHERE candidate_id = ?", (candidate_id,))
        
        # Then add the updated ones
        work_exp_count = int(request.form.get('work_exp_count', 0))
        for i in range(work_exp_count):
            company = request.form.get(f'company_{i}')
            position = request.form.get(f'position_{i}')
            start_date = request.form.get(f'start_date_{i}')
            end_date = request.form.get(f'end_date_{i}')
            description = request.form.get(f'description_{i}')
            
            if company and position:  # Only add if essential fields are present
                cursor.execute(
                    "INSERT INTO work_experience (candidate_id, company, position, start_date, end_date, description) VALUES (?, ?, ?, ?, ?, ?)",
                    (candidate_id, company, position, start_date, end_date, description)
                )
        
        # Handle skills updates
        # First, delete existing skills
        cursor.execute("DELETE FROM skills WHERE candidate_id = ?", (candidate_id,))
        
        # Insert updated skills
        skills_count = int(request.form.get('skills_count', 0))
        for i in range(skills_count):
            skill_name = request.form.get(f'skill_name_{i}')
            years_experience = request.form.get(f'years_experience_{i}')
            
            if skill_name:  # Only add if skill name is present
                try:
                    years_exp = float(years_experience) if years_experience else 0
                except ValueError:
                    years_exp = 0
                    
                cursor.execute(
                    "INSERT INTO skills (candidate_id, skill_name, years_experience) VALUES (?, ?, ?)",
                    (candidate_id, skill_name, years_exp)
                )
        
        # Process certificates
        certificates_count = int(request.form.get('certificates_count', 0))
        
        # First, delete existing certificates
        cursor.execute("DELETE FROM certificates WHERE candidate_id = ?", (candidate_id,))
        
        # Insert updated certificates
        for i in range(certificates_count):
            name = request.form.get(f'cert_name_{i}')
            if name:  # Only insert if there's at least a name
                cursor.execute("""
                    INSERT INTO certificates (candidate_id, name, issuer, date_obtained, 
                                           expiry_date, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    candidate_id,
                    name,
                    request.form.get(f'cert_issuer_{i}', ''),
                    request.form.get(f'cert_date_obtained_{i}', ''),
                    request.form.get(f'cert_expiry_date_{i}', ''),
                    request.form.get(f'cert_description_{i}', '')
                ))
        
        conn.commit()
        conn.close()
        
        flash('Candidate information updated successfully')
        return redirect(url_for('view_candidate', candidate_id=candidate_id))
    
    # GET request - show edit form
    conn = sqlite3.connect('cv_database.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get candidate info
    cursor.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,))
    candidate = cursor.fetchone()
    
    # Get work experience
    cursor.execute("SELECT * FROM work_experience WHERE candidate_id = ?", (candidate_id,))
    work_experience = cursor.fetchall()
    
    # Get skills
    cursor.execute("SELECT * FROM skills WHERE candidate_id = ?", (candidate_id,))
    skills = cursor.fetchall()
    
    # Get certificates
    cursor.execute("SELECT * FROM certificates WHERE candidate_id = ? ORDER BY date_obtained DESC", (candidate_id,))
    certificates = cursor.fetchall()
    
    conn.close()
    
    return render_template('edit_candidate.html', 
                          candidate=candidate, 
                          work_experience=work_experience, 
                          skills=skills,
                          certificates=certificates)

@app.route('/vacancy-match', methods=['GET', 'POST'])
def vacancy_match():
    # Initialize empty skills structure
    vacancy_skills = {
        'required_skills': [],
        'nice_to_have_skills': []
    }
    
    if request.method == 'POST':
        # Check if it's an AJAX request
        if request.is_json:
            data = request.get_json()
            vacancy_text = data.get('vacancy_text', '')
            vacancy_skills = {
                'required_skills': data.get('required_skills', []),
                'nice_to_have_skills': data.get('nice_to_have_skills', [])
            }
            
            # Get all candidates and their skills from the database
            conn = sqlite3.connect('cv_database.db')
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    c.id, 
                    c.name, 
                    c.email,
                    GROUP_CONCAT(s.skill_name || ':' || s.years_experience) as skills
                FROM candidates c
                LEFT JOIN skills s ON c.id = s.candidate_id
                GROUP BY c.id
            """)
            candidates = cursor.fetchall()
            
            # Calculate match percentage for each candidate
            matches = []
            for candidate in candidates:
                candidate_skills = {}
                if candidate['skills']:
                    for skill_entry in candidate['skills'].split(','):
                        if ':' in skill_entry:
                            skill_name, years = skill_entry.split(':', 1)
                            try:
                                candidate_skills[skill_name] = float(years)
                            except ValueError:
                                candidate_skills[skill_name] = 0
                
                required_match_score = 0
                required_total = len(vacancy_skills['required_skills'])
                nice_to_have_match_score = 0
                nice_to_have_total = len(vacancy_skills['nice_to_have_skills'])
                
                details = []
                
                # Check required skills
                for req_skill in vacancy_skills['required_skills']:
                    skill_name = req_skill['skill_name']
                    req_years = req_skill['years_experience']
                    
                    best_match = None
                    best_score = 0
                    best_years = 0
                    
                    for cand_skill, cand_years in candidate_skills.items():
                        similarity = skill_similarity(skill_name, cand_skill)
                        if similarity > 0.5 and similarity > best_score:  # Threshold of 0.5 for similarity
                            best_match = cand_skill
                            best_score = similarity
                            best_years = cand_years
                    
                    if best_match:
                        if best_years >= req_years:
                            required_match_score += 1
                            details.append({
                                'skill_name': skill_name,
                                'type': 'Required',
                                'match': True,
                                'reason': f"Candidate has {best_years} years of {best_match}, required {req_years} years."
                            })
                        else:
                            required_match_score += 0.5
                            details.append({
                                'skill_name': skill_name,
                                'type': 'Required',
                                'match': False,
                                'reason': f"Candidate has {best_years} years of {best_match}, required {req_years} years."
                            })
                    else:
                        details.append({
                            'skill_name': skill_name,
                            'type': 'Required',
                            'match': False,
                            'reason': "Skill not found in candidate's profile."
                        })
                
                # Check nice-to-have skills
                for nice_skill in vacancy_skills['nice_to_have_skills']:
                    skill_name = nice_skill['skill_name']
                    req_years = nice_skill['years_experience']
                    
                    match_found = False
                    for cand_skill, cand_years in candidate_skills.items():
                        # Simple matching for now, can be enhanced with fuzzy matching
                        if skill_name.lower() in cand_skill.lower() or cand_skill.lower() in skill_name.lower():
                            match_found = True
                            if cand_years >= req_years:
                                nice_to_have_match_score += 1
                                details.append({
                                    'skill_name': skill_name,
                                    'type': 'Nice to Have',
                                    'match': True,
                                    'reason': f"Candidate has {cand_years} years, preferred {req_years} years."
                                })
                            else:
                                nice_to_have_match_score += 0.5
                                details.append({
                                    'skill_name': skill_name,
                                    'type': 'Nice to Have',
                                    'match': False,
                                    'reason': f"Candidate has {cand_years} years, preferred {req_years} years."
                                })
                            break
                    
                    if not match_found:
                        details.append({
                            'skill_name': skill_name,
                            'type': 'Nice to Have',
                            'match': False,
                            'reason': "Skill not found in candidate's profile."
                        })
                
                # Calculate percentages
                required_percentage = (required_match_score / required_total * 100) if required_total > 0 else 0
                nice_to_have_percentage = (nice_to_have_match_score / nice_to_have_total * 100) if nice_to_have_total > 0 else 0
                
                # Calculate total match percentage with weighted formula
                required_weight = 0.7  # 70% weight for required skills
                nice_to_have_weight = 0.3  # 30% weight for nice-to-have skills
                
                # If there are no nice-to-have skills, give 100% weight to required skills
                if nice_to_have_total == 0:
                    match_percentage = required_percentage
                # If there are no required skills, give 100% weight to nice-to-have skills
                elif required_total == 0:
                    match_percentage = nice_to_have_percentage
                # Otherwise, use the weighted formula
                else:
                    match_percentage = (required_percentage * required_weight) + (nice_to_have_percentage * nice_to_have_weight)
                
                # Round to nearest integer
                match_percentage = round(match_percentage)
                required_percentage = round(required_percentage)
                nice_to_have_percentage = round(nice_to_have_percentage)
                
                # Add the match to the results
                matches.append({
                    'candidate_id': candidate['id'],
                    'name': candidate['name'],
                    'email': candidate['email'],
                    'match_percentage': match_percentage,
                    'required_match': required_percentage,
                    'nice_to_have_match': nice_to_have_percentage,
                    'details': details
                })
            
            conn.close()
            
            # Sort matches by percentage
            matches.sort(key=lambda x: x['match_percentage'], reverse=True)
            
            # Return JSON data instead of HTML
            return jsonify({
                'matches': matches
            })
        
        # Handle regular form submission
        vacancy_text = request.form.get('vacancy_text')
        
        # Extract skills from vacancy using Azure OpenAI
        prompt = f"""
        Extract required skills and experience from this job vacancy.
        Format the output as JSON with the following structure:
        {{
            "required_skills": [
                {{
                    "skill_name": "skill name",
                    "years_experience": minimum years required (number, 0 if not specified)
                }}
            ],
            "nice_to_have_skills": [
                {{
                    "skill_name": "skill name",
                    "years_experience": minimum years preferred (number, 0 if not specified)
                }}
            ]
        }}
        
        Vacancy Text:
        {vacancy_text}
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured information from job vacancies. Try to extract the most important skills and experience from the vacancy and translate to Dutch."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        vacancy_skills = json.loads(response.choices[0].message.content)
        
        # Get all candidates and their skills from the database
        conn = sqlite3.connect('cv_database.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                c.id, 
                c.name, 
                c.email,
                GROUP_CONCAT(s.skill_name || ':' || s.years_experience) as skills
            FROM candidates c
            LEFT JOIN skills s ON c.id = s.candidate_id
            GROUP BY c.id
        """)
        candidates = cursor.fetchall()
        
        # Calculate match percentage for each candidate
        matches = []
        for candidate in candidates:
            candidate_skills = {}
            if candidate['skills']:
                for skill_entry in candidate['skills'].split(','):
                    if ':' in skill_entry:
                        skill_name, years = skill_entry.split(':', 1)
                        try:
                            candidate_skills[skill_name] = float(years)
                        except ValueError:
                            candidate_skills[skill_name] = 0
            
            required_match_score = 0
            required_total = len(vacancy_skills['required_skills'])
            nice_to_have_match_score = 0
            nice_to_have_total = len(vacancy_skills['nice_to_have_skills'])
            
            details = []
            
            # Check required skills
            for req_skill in vacancy_skills['required_skills']:
                skill_name = req_skill['skill_name']
                req_years = req_skill['years_experience']
                
                best_match = None
                best_score = 0
                best_years = 0
                
                for cand_skill, cand_years in candidate_skills.items():
                    similarity = skill_similarity(skill_name, cand_skill)
                    if similarity > 0.5 and similarity > best_score:  # Threshold of 0.5 for similarity
                        best_match = cand_skill
                        best_score = similarity
                        best_years = cand_years
                
                if best_match:
                    if best_years >= req_years:
                        required_match_score += 1
                        details.append({
                            'skill_name': skill_name,
                            'type': 'Required',
                            'match': True,
                            'reason': f"Candidate has {best_years} years of {best_match}, required {req_years} years."
                        })
                    else:
                        required_match_score += 0.5
                        details.append({
                            'skill_name': skill_name,
                            'type': 'Required',
                            'match': False,
                            'reason': f"Candidate has {best_years} years of {best_match}, required {req_years} years."
                        })
                else:
                    details.append({
                        'skill_name': skill_name,
                        'type': 'Required',
                        'match': False,
                        'reason': "Skill not found in candidate's profile."
                    })
            
            # Check nice-to-have skills
            for nice_skill in vacancy_skills['nice_to_have_skills']:
                skill_name = nice_skill['skill_name']
                req_years = nice_skill['years_experience']
                
                match_found = False
                for cand_skill, cand_years in candidate_skills.items():
                    # Simple matching for now, can be enhanced with fuzzy matching
                    if skill_name.lower() in cand_skill.lower() or cand_skill.lower() in skill_name.lower():
                        match_found = True
                        if cand_years >= req_years:
                            nice_to_have_match_score += 1
                            details.append({
                                'skill_name': skill_name,
                                'type': 'Nice to Have',
                                'match': True,
                                'reason': f"Candidate has {cand_years} years, preferred {req_years} years."
                            })
                        else:
                            nice_to_have_match_score += 0.5
                            details.append({
                                'skill_name': skill_name,
                                'type': 'Nice to Have',
                                'match': False,
                                'reason': f"Candidate has {cand_years} years, preferred {req_years} years."
                            })
                        break
                
                if not match_found:
                    details.append({
                        'skill_name': skill_name,
                        'type': 'Nice to Have',
                        'match': False,
                        'reason': "Skill not found in candidate's profile."
                    })
            
            # Calculate percentages
            required_percentage = (required_match_score / required_total * 100) if required_total > 0 else 0
            nice_to_have_percentage = (nice_to_have_match_score / nice_to_have_total * 100) if nice_to_have_total > 0 else 0
            
            # Calculate total match percentage with weighted formula
            required_weight = 0.7  # 70% weight for required skills
            nice_to_have_weight = 0.3  # 30% weight for nice-to-have skills
            
            # If there are no nice-to-have skills, give 100% weight to required skills
            if nice_to_have_total == 0:
                match_percentage = required_percentage
            # If there are no required skills, give 100% weight to nice-to-have skills
            elif required_total == 0:
                match_percentage = nice_to_have_percentage
            # Otherwise, use the weighted formula
            else:
                match_percentage = (required_percentage * required_weight) + (nice_to_have_percentage * nice_to_have_weight)
            
            # Round to nearest integer
            match_percentage = round(match_percentage)
            required_percentage = round(required_percentage)
            nice_to_have_percentage = round(nice_to_have_percentage)
            
            # Add the match to the results
            matches.append({
                'candidate_id': candidate['id'],
                'name': candidate['name'],
                'email': candidate['email'],
                'match_percentage': match_percentage,
                'required_match': required_percentage,
                'nice_to_have_match': nice_to_have_percentage,
                'details': details
            })
        
        conn.close()
        
        # Sort matches by percentage
        matches.sort(key=lambda x: x['match_percentage'], reverse=True)
        
        # Return full page for form submissions
        return render_template(
            'vacancy_match.html',
            vacancy_text=vacancy_text,
            vacancy_skills=vacancy_skills,
            matches=matches,
            show_results=True
        )
    
    return render_template('vacancy_match.html', show_results=False)

def skill_similarity(skill1, skill2):
    """Calculate similarity between two skill names"""
    s1 = skill1.lower()
    s2 = skill2.lower()
    
    # Direct match or substring match
    if s1 == s2 or s1 in s2 or s2 in s1:
        return 1.0
    
    # Check for word-level matches
    words1 = set(s1.split())
    words2 = set(s2.split())
    common_words = words1.intersection(words2)
    
    if common_words:
        return len(common_words) / max(len(words1), len(words2))
    
    return 0.0

@app.route('/skill-search', methods=['GET', 'POST'])
def skill_search():
    skill = None
    candidates = []
    
    if request.method == 'POST':
        skill = request.form.get('skill', '').strip()
    elif request.method == 'GET':
        skill = request.args.get('skill', '').strip()
    
    if skill:
        # Connect to the database
        conn = sqlite3.connect('cv_database.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get candidates with the specified skill
        cursor.execute("""
            SELECT 
                c.id, 
                c.name, 
                c.email, 
                c.phone,
                s.years_experience
            FROM candidates c
            JOIN skills s ON c.id = s.candidate_id
            WHERE LOWER(s.skill_name) = LOWER(?)
            ORDER BY s.years_experience DESC
        """, (skill,))
        
        skill_candidates = cursor.fetchall()
        
        # For each candidate, get their other skills
        for candidate in skill_candidates:
            cursor.execute("""
                SELECT skill_name, years_experience
                FROM skills
                WHERE candidate_id = ? AND LOWER(skill_name) != LOWER(?)
                ORDER BY years_experience DESC
            """, (candidate['id'], skill))
            
            other_skills = cursor.fetchall()
            
            # Convert to a dictionary and add other skills
            candidate_dict = dict(candidate)
            candidate_dict['other_skills'] = [dict(s) for s in other_skills]
            candidates.append(candidate_dict)
        
        conn.close()
    
    return render_template('skill_search.html', skill=skill, candidates=candidates)

def get_candidate_by_id(candidate_id):
    conn = sqlite3.connect('cv_database.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get candidate info
    cursor.execute('''
        SELECT c.*, GROUP_CONCAT(s.skill_name) as skills_list
        FROM candidates c
        LEFT JOIN skills s ON c.id = s.candidate_id
        WHERE c.id = ?
        GROUP BY c.id
    ''', (candidate_id,))
    
    candidate = cursor.fetchone()
    
    if candidate:
        # Get skills separately to maintain the full skill information
        cursor.execute('''
            SELECT skill_name, years_experience
            FROM skills 
            WHERE candidate_id = ?
        ''', (candidate_id,))
        skills = cursor.fetchall()
        
        # Convert candidate to dict and add skills
        candidate_dict = dict(candidate)
        candidate_dict['skills'] = [
            {
                'name': skill['skill_name'],
                'years': skill['years_experience']
            } for skill in skills
        ]
        
        # Get work experience
        cursor.execute('''
            SELECT company, position, start_date, end_date, description
            FROM work_experience
            WHERE candidate_id = ?
            ORDER BY start_date DESC
        ''', (candidate_id,))
        work_experience = cursor.fetchall()
        candidate_dict['work_experience'] = [dict(exp) for exp in work_experience]
        
        # Get certificates
        cursor.execute('''
            SELECT name, issuer, date_obtained, expiry_date, description
            FROM certificates
            WHERE candidate_id = ?
            ORDER BY date_obtained DESC
        ''', (candidate_id,))
        certificates = cursor.fetchall()
        candidate_dict['certificates'] = [dict(cert) for cert in certificates]
        
        conn.close()
        return candidate_dict
    
    conn.close()
    return None

@app.route('/write-cover-letter/<string:candidate_id>')
def write_cover_letter(candidate_id):
    candidate = get_candidate_by_id(candidate_id)
    if not candidate:
        flash('Kandidaat niet gevonden', 'error')
        return redirect(url_for('vacancy_match'))
        
    return render_template(
        'cover_letter_writer.html',
        candidate=candidate,
        vacancy_text=request.args.get('vacancy_text', '')
    )

@app.route('/generate-cover-letter', methods=['POST'])
def generate_cover_letter():
    data = request.get_json()
    vacancy_text = data['vacancy_text']
    candidate_id = data['candidate_id']
    language = data['language']
    
    # Get candidate information
    candidate = get_candidate_by_id(candidate_id)
    
    # Create prompt for Azure OpenAI
    prompt = f"""
    Please write a professional cover letter in {'Dutch' if language == 'nl' else 'English'} for the following job:
    
    Vacancy:
    {vacancy_text}
    
    Candidate Information:
    Name: {candidate['name']}
    Skills: {', '.join(skill['name'] for skill in candidate['skills'])}
    Work Experience: {', '.join(exp['position'] + ' at ' + exp['company'] for exp in candidate['work_experience'])}
    
    Please write a personalized cover letter that:
    1. Matches the candidate's experience with the job requirements
    2. Highlights relevant skills and experience
    3. Uses a professional but engaging tone
    4. Is structured with proper paragraphs
    5. Is in {'Dutch' if language == 'nl' else 'English'}
    """
    
    # Use your existing Azure OpenAI client
    response = client.chat.completions.create(
        model="gpt-4",  # or whatever model you're using
        messages=[
            {"role": "system", "content": "You are a professional cover letter writer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    cover_letter = response.choices[0].message.content
    
    return jsonify({'cover_letter': cover_letter})

@app.route('/generate-candidate-cv/<string:candidate_id>')
def generate_candidate_cv(candidate_id):
    try:
        # Call the generate_cv function from cv_generator.py
        from cv_generator import generate_cv
        
        # Generate the CV
        html_content = generate_cv(candidate_id)
        
        # Create a unique filename
        filename = f"cv_{candidate_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.html"
        
        # Create a response with the HTML content
        response = make_response(html_content)
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["Content-Type"] = "text/html"
        
        return response
    except Exception as e:
        flash(f'Error generating CV: {str(e)}', 'error')
        return redirect(url_for('view_candidate', candidate_id=candidate_id))

@app.route('/view_cv/<candidate_id>')
def view_cv(candidate_id):
    try:
        # Generate the CV HTML
        html_content = generate_cv(candidate_id)
        
        # Return the HTML content
        return html_content
    except Exception as e:
        import traceback
        print(f"Error generating CV: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)