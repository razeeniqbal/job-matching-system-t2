import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
import joblib
import os

# Download NLTK data - This will automatically download if missing
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')

class SkillsFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts features related to skill match between job requirements and talent profile.
    Uses semantic matching with TF-IDF vectors.
    """
    
    def __init__(self, use_semantic_match=True):
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=self._tokenize,
            stop_words='english',
            max_features=1000
        )
        self.use_semantic_match = use_semantic_match
        self.lemmatizer = WordNetLemmatizer()
        
    def _tokenize(self, text):
        # Lemmatize and tokenize text
        return [self.lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(text)]
    
    def fit(self, X, y=None):
        # Create a corpus of all skills to fit the vectorizer
        all_skills = []
        for data in X:
            all_skills.extend(data.get('job_requirements', {}).get('core_skills', []))
            all_skills.extend(data.get('talent_profile', {}).get('skills', []))
        
        skill_corpus = [' '.join(skills) for skills in [all_skills]]
        self.vectorizer.fit(skill_corpus)
        return self
    
    def transform(self, X):
        result = []
        
        for data in X:
            job_skills = data.get('job_requirements', {}).get('core_skills', [])
            talent_skills = data.get('talent_profile', {}).get('skills', [])
            
            # Calculate basic coverage
            required_skills_count = len(job_skills)
            matching_skills_count = sum(1 for skill in job_skills if skill in talent_skills)
            skill_coverage = matching_skills_count / required_skills_count if required_skills_count > 0 else 0
            
            # Calculate additional skills
            additional_skills = len([skill for skill in talent_skills if skill not in job_skills])
            
            if self.use_semantic_match and job_skills and talent_skills:
                # Calculate semantic similarity using TF-IDF
                job_skills_text = ' '.join(job_skills)
                talent_skills_text = ' '.join(talent_skills)
                
                job_vector = self.vectorizer.transform([job_skills_text])
                talent_vector = self.vectorizer.transform([talent_skills_text])
                
                semantic_similarity = cosine_similarity(job_vector, talent_vector)[0][0]
            else:
                semantic_similarity = 0
            
            result.append({
                'skill_coverage': skill_coverage,
                'semantic_similarity': semantic_similarity,
                'matching_skills_count': matching_skills_count,
                'additional_skills': additional_skills,
                'total_skills': len(talent_skills)
            })
            
        return pd.DataFrame(result)

class ExperienceFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts features related to work experience.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        result = []
        
        for data in X:
            required_exp = data.get('job_requirements', {}).get('min_experience', 0)
            talent_exp = data.get('talent_profile', {}).get('experience', 0)
            
            # Experience match ratio (clipped at 2.0)
            exp_ratio = min(talent_exp / required_exp if required_exp > 0 else 2.0, 2.0)
            
            # Experience difference (positive if talent exceeds requirement)
            exp_diff = talent_exp - required_exp
            
            # Is experience sufficient?
            meets_exp_req = 1 if talent_exp >= required_exp else 0
            
            result.append({
                'experience_ratio': exp_ratio,
                'experience_difference': exp_diff,
                'meets_experience_requirement': meets_exp_req
            })
            
        return pd.DataFrame(result)

class EducationFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts features related to education level match.
    """
    
    def __init__(self):
        self.education_hierarchy = {
            'High School': 1,
            'Diploma': 2,
            'Associate': 3,
            'Bachelor\'s': 4,
            'Master\'s': 5,
            'PhD': 6
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        result = []
        
        for data in X:
            required_edu = data.get('job_requirements', {}).get('education', 'Bachelor\'s')
            talent_edu = data.get('talent_profile', {}).get('education', 'High School')
            
            required_level = self.education_hierarchy.get(required_edu, 4)  # Default to Bachelor's
            talent_level = self.education_hierarchy.get(talent_edu, 1)  # Default to High School
            
            # Education level difference
            edu_diff = talent_level - required_level
            
            # Education meets requirement
            meets_edu_req = 1 if talent_level >= required_level else 0
            
            # Exceeds requirement by levels
            exceeds_by = max(0, edu_diff)
            
            result.append({
                'education_difference': edu_diff,
                'meets_education_requirement': meets_edu_req,
                'exceeds_education_by': exceeds_by
            })
            
        return pd.DataFrame(result)

class CertificationFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts features related to certification match.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        result = []
        
        for data in X:
            required_certs = data.get('job_requirements', {}).get('certifications', [])
            talent_certs = data.get('talent_profile', {}).get('certifications', [])
            
            # Count matching certifications
            matching_certs = sum(1 for cert in required_certs if cert in talent_certs)
            
            # Certification coverage
            cert_coverage = matching_certs / len(required_certs) if required_certs else 1.0
            
            # Additional certifications
            additional_certs = len([cert for cert in talent_certs if cert not in required_certs])
            
            # Has all required certifications
            has_all_certs = 1 if matching_certs == len(required_certs) and required_certs else 0
            
            result.append({
                'certification_coverage': cert_coverage,
                'matching_certifications': matching_certs,
                'additional_certifications': additional_certs,
                'has_all_required_certifications': has_all_certs
            })
            
        return pd.DataFrame(result)

class FeatureEngineeringPipeline:
    """
    Main feature engineering pipeline that combines all feature extractors.
    """
    
    def __init__(self, use_semantic_match=True):
        self.pipeline_components = {
            'skills': SkillsFeatureExtractor(use_semantic_match),
            'experience': ExperienceFeatureExtractor(),
            'education': EducationFeatureExtractor(),
            'certifications': CertificationFeatureExtractor()
        }
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        # Fit all feature extractors
        transformed_data = {}
        for name, extractor in self.pipeline_components.items():
            transformed_data[name] = extractor.fit_transform(X)
            
        # Combine all features
        combined_features = pd.concat(transformed_data.values(), axis=1)
        
        # Fit the scaler
        self.scaler.fit(combined_features)
        
        return self
    
    def transform(self, X):
        # Transform all features
        transformed_data = {}
        for name, extractor in self.pipeline_components.items():
            transformed_data[name] = extractor.transform(X)
            
        # Combine all features
        combined_features = pd.concat(transformed_data.values(), axis=1)
        
        # Scale the features
        scaled_features = self.scaler.transform(combined_features)
        
        return scaled_features
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    def save(self, file_path='models/feature_pipeline.joblib'):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)
    
    @classmethod
    def load(cls, file_path='models/feature_pipeline.joblib'):
        return joblib.load(file_path)

def get_raw_match_scores(X):
    """
    Calculate raw match scores based on the weighted criteria without ML.
    Useful for comparison with ML predictions.
    """
    results = []
    
    weights = {
        'core_skills': 0.5,  # 50%
        'experience': 0.2,   # 20%
        'education': 0.15,   # 15%
        'certifications': 0.15  # 15%
    }
    
    education_hierarchy = {
        'High School': 1,
        'Diploma': 2,
        'Associate': 3,
        'Bachelor\'s': 4,
        'Master\'s': 5,
        'PhD': 6
    }
    
    for data in X:
        job_app_id = data.get('job_app_id')
        job_req = data.get('job_requirements', {})
        talent = data.get('talent_profile', {})
        
        # Core Skills Score
        required_skills = job_req.get('core_skills', [])
        talent_skills = talent.get('skills', [])
        skill_score = sum(1 for skill in required_skills if skill in talent_skills) / len(required_skills) if required_skills else 0
        
        # Experience Score
        required_exp = job_req.get('min_experience', 0)
        talent_exp = talent.get('experience', 0)
        exp_score = min(talent_exp / required_exp if required_exp > 0 else 1.0, 1.0)
        
        # Education Score
        required_edu = job_req.get('education', 'Bachelor\'s')
        talent_edu = talent.get('education', 'High School')
        req_edu_level = education_hierarchy.get(required_edu, 4)
        talent_edu_level = education_hierarchy.get(talent_edu, 1)
        edu_score = min(talent_edu_level / req_edu_level, 1.0)
        
        # Certification Score
        required_certs = job_req.get('certifications', [])
        talent_certs = talent.get('certifications', [])
        cert_score = sum(1 for cert in required_certs if cert in talent_certs) / len(required_certs) if required_certs else 1.0
        
        # Weighted Score
        weighted_score = (
            skill_score * weights['core_skills'] +
            exp_score * weights['experience'] +
            edu_score * weights['education'] +
            cert_score * weights['certifications']
        ) * 100
        
        results.append({
            'job_app_id': job_app_id,
            'score': round(weighted_score, 2)
        })
    
    return results