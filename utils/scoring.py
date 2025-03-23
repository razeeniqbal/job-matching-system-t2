# utils/scoring.py
from config.settings import EDUCATION_HIERARCHY

def get_score_color(score):
    """Return color based on score range."""
    if score >= 90:
        return '#4CAF50'  # Green
    elif score >= 70:
        return '#2196F3'  # Blue
    elif score >= 50:
        return '#FFC107'  # Amber
    else:
        return '#F44336'  # Red

def calculate_component_scores(job_requirements, talent_profile):
    """Calculate individual component scores for visualization."""
    # Skills match
    job_skills = job_requirements.get('core_skills', [])
    talent_skills = talent_profile.get('skills', [])
    skill_match = sum(1 for skill in job_skills if skill in talent_skills) / len(job_skills) if job_skills else 1.0
    
    # Experience match
    min_experience = job_requirements.get('min_experience', 0)
    experience = talent_profile.get('experience', 0)
    exp_match = min(experience / min_experience, 1.0) if min_experience > 0 else 1.0
    
    # Education match
    required_edu = job_requirements.get('education', 'Bachelor\'s')
    talent_edu = talent_profile.get('education', 'High School')
    req_edu_level = EDUCATION_HIERARCHY.get(required_edu, 3)  # Default to Bachelor's
    talent_edu_level = EDUCATION_HIERARCHY.get(talent_edu, 1)  # Default to High School
    edu_match = min(talent_edu_level / req_edu_level, 1.0) if req_edu_level > 0 else 1.0
    
    # Certification match
    job_certs = job_requirements.get('certifications', [])
    talent_certs = talent_profile.get('certifications', [])
    cert_match = sum(1 for cert in job_certs if cert in talent_certs) / len(job_certs) if job_certs else 1.0
    
    return {
        'skill_match': skill_match,
        'exp_match': exp_match,
        'edu_match': edu_match,
        'cert_match': cert_match
    }