import sqlite3
import os
from datetime import datetime
import uuid
import re
import logging

def generate_cv(candidate_id):
    """
    Generate an HTML CV for the specified candidate by pulling data from the database
    and filling in the HTML template.
    
    Args:
        candidate_id (str): The ID of the candidate to generate the CV for
    
    Returns:
        str: The generated HTML content
    """
    # Connect to the database
    conn = sqlite3.connect('cv_database.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Fetch candidate basic information
    cursor.execute('''
        SELECT name, email, phone, cv_text
        FROM candidates
        WHERE id = ?
    ''', (candidate_id,))
    
    candidate = cursor.fetchone()
    if not candidate:
        raise ValueError(f"Candidate with ID {candidate_id} not found")
    
    # Fetch skills
    cursor.execute('''
        SELECT skill_name, years_experience
        FROM skills
        WHERE candidate_id = ?
        ORDER BY years_experience DESC
    ''', (candidate_id,))
    skills = cursor.fetchall()
    
    # Fetch work experience
    cursor.execute('''
        SELECT company, position, start_date, end_date, description
        FROM work_experience
        WHERE candidate_id = ?
        ORDER BY start_date DESC
    ''', (candidate_id,))
    work_experiences = cursor.fetchall()
    
    # Fetch certificates
    cursor.execute('''
        SELECT name, issuer, date_obtained, expiry_date, description
        FROM certificates
        WHERE candidate_id = ?
        ORDER BY date_obtained DESC
    ''', (candidate_id,))
    certificates = cursor.fetchall()
    
    # Close the database connection
    conn.close()
    
    # Parse the CV text to extract education information (assuming it's in the CV text)
    education_info = extract_education_from_cv(candidate['cv_text'])
    
    # Create the HTML content
    html_content = generate_html_cv(
        candidate=candidate,
        work_experiences=work_experiences,
        skills=skills,
        certificates=certificates,
        education=education_info
    )
    
    return html_content

def extract_education_from_cv(cv_text):
    """
    Extract education information from the CV text.
    This is a simple implementation and might need refinement based on CV formats.
    
    Args:
        cv_text (str): The full CV text
    
    Returns:
        list: A list of dictionaries containing education information
    """
    # This is a simplified extraction. In reality, you might want to use NLP or more sophisticated regex
    education = []
    
    # Example regex patterns (very simplified)
    education_pattern = r"(?i)(Bachelor|Master|PhD|BSc|MSc|MBA|Doctorate)[^\n]+(University|College|Institute)[^\n]+(\d{4})\s*-\s*(\d{4}|\bpresent\b)"
    
    matches = re.finditer(education_pattern, cv_text)
    for match in matches:
        education_text = match.group(0).strip()
        parts = education_text.split(',')
        
        # This is very simplified - you would need more robust parsing
        degree = parts[0].strip() if len(parts) > 0 else ""
        institution = parts[1].strip() if len(parts) > 1 else ""
        
        # Look for years
        years_match = re.search(r"(\d{4})\s*-\s*(\d{4}|\bpresent\b)", education_text)
        if years_match:
            start_year = years_match.group(1)
            end_year = years_match.group(2)
        else:
            start_year = ""
            end_year = ""
            
        education.append({
            "degree": degree,
            "institution": institution,
            "start_year": start_year,
            "end_year": end_year,
            "description": ""  # No description extraction in this simplified version
        })
    
    return education

def generate_html_cv(candidate, work_experiences, skills, certificates, education):
    """
    Generate the HTML content for the CV.
    
    Args:
        candidate (dict): Basic candidate information
        work_experiences (list): List of work experiences
        skills (list): List of skills
        certificates (list): List of certificates
        education (list): List of education entries
    
    Returns:
        str: The complete HTML content for the CV
    """
    # Group skills by type (this is a simplified approach)
    programming_skills = []
    other_skills = []
    
    programming_keywords = ['python', 'java', 'javascript', 'html', 'css', 'sql', 'c++', 'c#', 'typescript', 'php']
    
    for skill in skills:
        if any(keyword in skill['skill_name'].lower() for keyword in programming_keywords):
            programming_skills.append(skill)
        else:
            other_skills.append(skill)
    
    # Calculate skill percentages based on years of experience
    max_years = max([skill['years_experience'] for skill in skills], default=1)
    
    # Generate the work experience HTML
    work_experience_html = ""
    for exp in work_experiences:
        current_position = exp['end_date'].lower() in ['', 'current', 'present']
        period = f"{exp['start_date']} - {'Present' if current_position else exp['end_date']}"
        
        description_points = exp['description'].split('\n')
        description_html = "<ul>"
        for point in description_points:
            if point.strip():
                description_html += f"<li>{point.strip()}</li>"
        description_html += "</ul>"
        
        work_experience_html += f'''
        <div class="timeline-item">
            <div class="job-title">{exp['position']}</div>
            <div class="company">{exp['company']}</div>
            <div class="period">{period}</div>
            <div class="job-description">
                {description_html}
            </div>
        </div>
        '''
    
    # Generate the education HTML
    education_html = ""
    for edu in education:
        period = f"{edu['start_year']} - {edu['end_year']}"
        
        education_html += f'''
        <div class="timeline-item">
            <div class="education-title">{edu['degree']}</div>
            <div class="institution">{edu['institution']}</div>
            <div class="period">{period}</div>
            <div class="education-description">
                <p>{edu['description']}</p>
            </div>
        </div>
        '''
    
    # Generate the skills HTML
    programming_skills_html = ""
    for skill in programming_skills:
        percentage = min(int((skill['years_experience'] / max_years) * 100), 100)
        programming_skills_html += f'''
        <div class="skill-item">
            <div class="skill-name">{skill['skill_name']} ({skill['years_experience']} years)</div>
            <div class="skill-bar">
                <div class="skill-level" style="width: {percentage}%;"></div>
            </div>
        </div>
        '''
    
    other_skills_html = ""
    for skill in other_skills:
        percentage = min(int((skill['years_experience'] / max_years) * 100), 100)
        other_skills_html += f'''
        <div class="skill-item">
            <div class="skill-name">{skill['skill_name']} ({skill['years_experience']} years)</div>
            <div class="skill-bar">
                <div class="skill-level" style="width: {percentage}%;"></div>
            </div>
        </div>
        '''
    
    # Generate the certificates HTML
    certificates_html = ""
    if certificates:
        certificates_html = '''
        <section id="certificates">
            <h2 class="section-title">Certificates</h2>
            <div class="certificates-list">
        '''
        
        for cert in certificates:
            expiry = f" (Valid until {cert['expiry_date']})" if cert['expiry_date'] else ""
            certificates_html += f'''
            <div class="certificate-item">
                <div class="certificate-name">{cert['name']}{expiry}</div>
                <div class="certificate-issuer">Issued by {cert['issuer']} - {cert['date_obtained']}</div>
                <div class="certificate-description">{cert['description']}</div>
            </div>
            '''
        
        certificates_html += '''
            </div>
        </section>
        '''
    
    # Generate the full HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{candidate['name']} - CV</title>
    <style>
        :root {{
            --primary-color: #00D66C;
            --primary-dark: #00b85d;
            --primary-light: #33db84;
            --secondary-color: #6C1F85;
            --secondary-dark: #591a6d;
            --secondary-light: #8229a0;
            --light-bg: #f8f9fa;
            --dark-bg: #212529;
            --text-color: #2c2c2c;
            --light-text: #f8f9fa;
            --border-radius: 8px;
            --box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            --transition: all 0.3s ease;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            transition: var(--transition);
        }}
        
        body {{
            color: var(--text-color);
            line-height: 1.6;
            background-color: var(--light-bg);
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background-color: white;
            box-shadow: var(--box-shadow);
            border-radius: var(--border-radius);
            overflow: hidden;
        }}
        
        header {{
            background-color: var(--primary-color);
            color: var(--light-text);
            padding: 40px 50px;
            position: relative;
        }}
        
        .header-content {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .profile-picture {{
            width: 150px;
            height: 150px;
            border-radius: 50%;
            border: 4px solid white;
            object-fit: cover;
            box-shadow: var(--box-shadow);
        }}
        
        .header-text h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        
        .header-text h2 {{
            font-size: 1.5rem;
            font-weight: 400;
            margin-bottom: 20px;
        }}
        
        .contact-info {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}
        
        .contact-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .accent-bar {{
            height: 8px;
            background-color: var(--secondary-color);
        }}
        
        main {{
            padding: 50px;
        }}
        
        section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            color: var(--primary-color);
            font-size: 1.6rem;
            margin-bottom: 20px;
            position: relative;
            padding-bottom: 10px;
        }}
        
        .section-title::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 60px;
            height: 3px;
            background-color: var(--secondary-color);
        }}
        
        .timeline-item {{
            margin-bottom: 30px;
            position: relative;
            padding-left: 30px;
        }}
        
        .timeline-item::before {{
            content: '';
            position: absolute;
            left: 0;
            top: 8px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: var(--primary-color);
        }}
        
        .timeline-item::after {{
            content: '';
            position: absolute;
            left: 5px;
            top: 25px;
            width: 2px;
            height: calc(100% - 25px);
            background-color: var(--primary-light);
        }}
        
        .timeline-item:last-child::after {{
            display: none;
        }}
        
        .job-title, .education-title {{
            font-weight: 600;
            font-size: 1.2rem;
            color: var(--primary-dark);
            margin-bottom: 5px;
        }}
        
        .company, .institution {{
            font-weight: 500;
            color: var(--text-color);
        }}
        
        .period {{
            color: var(--secondary-color);
            font-weight: 500;
            margin-bottom: 10px;
        }}
        
        .job-description, .education-description {{
            margin-top: 10px;
        }}
        
        .skills-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}
        
        .skill-category {{
            flex: 1;
            min-width: 250px;
        }}
        
        .skill-category h3 {{
            color: var(--secondary-color);
            margin-bottom: 15px;
            font-size: 1.2rem;
        }}
        
        .skill-item {{
            margin-bottom: 10px;
        }}
        
        .skill-name {{
            margin-bottom: 5px;
            font-weight: 500;
        }}
        
        .skill-bar {{
            height: 8px;
            background-color: var(--light-bg);
            border-radius: var(--border-radius);
            overflow: hidden;
        }}
        
        .skill-level {{
            height: 100%;
            background-color: var(--primary-color);
        }}
        
        .certificates-list {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        
        .certificate-item {{
            border-left: 3px solid var(--primary-color);
            padding-left: 15px;
            margin-bottom: 15px;
        }}
        
        .certificate-name {{
            font-weight: 600;
            margin-bottom: 5px;
            color: var(--primary-dark);
        }}
        
        .certificate-issuer {{
            font-style: italic;
            margin-bottom: 5px;
        }}
        
        .languages-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}
        
        .language-item {{
            flex: 1;
            min-width: 120px;
        }}
        
        .interests-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .interest-item {{
            background-color: var(--light-bg);
            padding: 8px 15px;
            border-radius: 20px;
            display: inline-block;
            transition: var(--transition);
        }}
        
        .interest-item:hover {{
            background-color: var(--primary-light);
            color: var(--light-text);
            transform: translateY(-2px);
        }}
        
        footer {{
            background-color: var(--primary-color);
            color: var(--light-text);
            text-align: center;
            padding: 20px;
            font-size: 0.9rem;
        }}

        ul {{
            padding-left: 20px;
        }}

        li {{
            margin-bottom: 5px;
        }}

        @media (max-width: 768px) {{
            .header-content {{
                flex-direction: column;
                text-align: center;
            }}
            
            .profile-picture {{
                margin-bottom: 20px;
            }}
            
            .contact-info {{
                justify-content: center;
            }}
            
            main {{
                padding: 30px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-content">
                <img src="/api/placeholder/150/150" alt="Profile Picture" class="profile-picture">
                <div class="header-text">
                    <h1>{candidate['name']}</h1>
                    <h2>{work_experiences[0]['position'] if work_experiences else 'Professional'}</h2>
                    <div class="contact-info">
                        <div class="contact-item">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path>
                            </svg>
                            {candidate['phone']}
                        </div>
                        <div class="contact-item">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
                                <polyline points="22,6 12,13 2,6"></polyline>
                            </svg>
                            {candidate['email']}
                        </div>
                    </div>
                </div>
            </div>
        </header>
        <div class="accent-bar"></div>
        <main>
            <section id="profile">
                <h2 class="section-title">Profile</h2>
                <p>{extract_profile_summary(candidate['cv_text'])}</p>
            </section>
            
            <section id="experience">
                <h2 class="section-title">Professional Experience</h2>
                {work_experience_html}
            </section>
            
            <section id="education">
                <h2 class="section-title">Education</h2>
                {education_html}
            </section>
            
            <section id="skills">
                <h2 class="section-title">Skills</h2>
                
                <div class="skills-container">
                    <div class="skill-category">
                        <h3>Programming Languages</h3>
                        {programming_skills_html}
                    </div>
                    
                    <div class="skill-category">
                        <h3>Technologies & Frameworks</h3>
                        {other_skills_html}
                    </div>
                </div>
            </section>
            
            {certificates_html}
            
        </main>
        
        <footer>
            <p>© {datetime.now().year} {candidate['name']} - Professional CV</p>
        </footer>
    </div>
</body>
</html>'''

    return html

def extract_profile_summary(cv_text, max_words=100):
    """
    Extract or generate a profile summary from the CV text.
    
    Args:
        cv_text (str): The full CV text
        max_words (int): Maximum number of words for the summary
    
    Returns:
        str: A brief professional summary
    """
    # Look for common summary section indicators
    summary_patterns = [
        r"(?i)(?:profile|summary|about me|professional summary)[:\n]+(.*?)(?:\n\n|\n#|\Z)",
        r"(?i)(?:career objective|objective|professional objective)[:\n]+(.*?)(?:\n\n|\n#|\Z)"
    ]
    
    for pattern in summary_patterns:
        match = re.search(pattern, cv_text, re.DOTALL)
        if match:
            # Clean up the extracted text
            summary = match.group(1).strip()
            summary = re.sub(r'\s+', ' ', summary)  # Replace multiple spaces with single space
            
            # Limit to max_words
            words = summary.split()
            if len(words) > max_words:
                summary = ' '.join(words[:max_words]) + '...'
                
            return summary
    
    # If no summary found, return a generic one based on work experience and skills
    return "Experienced professional with a track record of success in relevant industries. Skilled in various technologies with a focus on delivering high-quality results."

def main():
    """
    Main function to demonstrate the CV generation.
    This would be replaced with your actual application logic.
    """
    # Example: Insert sample data and generate a CV
    candidate_id = insert_sample_data()
    cv_path = generate_cv('00bf0fb6-d358-46a4-a7ca-5abab120267f')
    print(f"CV generated successfully at: {cv_path}")

def insert_sample_data():
    """
    Insert sample data into the database for demonstration purposes.
    
    Returns:
        str: The ID of the inserted candidate
    """
    # Connect to the database
    conn = sqlite3.connect('candidates.db')
    cursor = conn.cursor()
    
    # Create a unique ID for the candidate
    candidate_id = str(uuid.uuid4())
    
    # Insert candidate
    cursor.execute('''
        INSERT INTO candidates (id, name, email, phone, cv_text)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        candidate_id,
        "Jane Smith",
        "jane.smith@example.com",
        "+31 6 12345678",
        """
Professional Summary:
Experienced software developer with 8+ years of experience in full-stack development. Skilled in designing and implementing scalable solutions for enterprise clients across various industries.

Education:
Master of Science in Computer Science, University of Amsterdam, 2012 - 2014
Bachelor of Science in Information Technology, Utrecht University, 2008 - 2012

Skills:
• Programming languages: Java, Python, JavaScript, TypeScript
• Frameworks: Spring, Django, React
• Cloud platforms: AWS, Azure
• Database: MySQL, PostgreSQL, MongoDB
• DevOps: Docker, Kubernetes, Jenkins
• Methodologies: Agile, Scrum, TDD
"""
    ))
    
    # Insert skills
    skills_data = [
        (candidate_id, "Java", 8.0),
        (candidate_id, "Python", 6.5),
        (candidate_id, "JavaScript", 7.0),
        (candidate_id, "TypeScript", 4.0),
        (candidate_id, "React", 5.0),
        (candidate_id, "Spring Framework", 6.0),
        (candidate_id, "AWS", 4.5),
        (candidate_id, "Docker", 3.0),
        (candidate_id, "Agile Methodologies", 7.0)
    ]
    
    cursor.executemany('''
        INSERT INTO skills (candidate_id, skill_name, years_experience)
        VALUES (?, ?, ?)
    ''', skills_data)
    
    # Insert work experience
    work_exp_data = [
        (
            candidate_id,
            "Tech Solutions B.V.",
            "Senior Software Developer",
            "2020-01",
            "Present",
            """Leading development of cloud-native applications
Architect and implement scalable solutions using microservices
Mentor junior developers and conduct code reviews
Implement CI/CD pipelines for automated testing and deployment"""
        ),
        (
            candidate_id,
            "Digital Innovations",
            "Software Developer",
            "2015-03",
            "2019-12",
            """Developed web applications for financial sector clients
Optimized database queries and application performance
Collaborated with UX designers to implement responsive interfaces
Participated in Agile development processes"""
        ),
        (
            candidate_id,
            "WebTech Solutions",
            "Junior Developer",
            "2012-09",
            "2015-02",
            """Assisted in development of front-end components
Fixed bugs and implemented feature enhancements
Participated in daily Scrum meetings"""
        )
    ]
    
    cursor.executemany('''
        INSERT INTO work_experience (candidate_id, company, position, start_date, end_date, description)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', work_exp_data)
    
    # Insert certificates
    certificates_data = [
        (
            candidate_id,
            "AWS Certified Solutions Architect",
            "Amazon Web Services",
            "2021-05",
            "2024-05",
            "Professional certification for designing distributed systems on AWS"
        ),
        (
            candidate_id,
            "Oracle Certified Professional: Java SE 11 Developer",
            "Oracle",
            "2019-10",
            "",
            "Advanced Java development certification"
        ),
        (
            candidate_id,
            "Professional Scrum Master I",
            "Scrum.org",
            "2018-03",
            "",
            "Certification in Scrum framework and Agile methodologies"
        )
    ]
    
    cursor.executemany('''
        INSERT INTO certificates (candidate_id, name, issuer, date_obtained, expiry_date, description)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', certificates_data)
    
    # Commit and close
    conn.commit()
    conn.close()
    
    return candidate_id

if __name__ == "__main__":
    main()