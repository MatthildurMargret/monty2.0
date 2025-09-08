
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add the parent directory to the Python path so we can import the services module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.openai_api import ask_monty

def generate_verticals(row, use_json=True):
    """
    Generate verticals for a founder's current company.

    Prefers JSON experience data:
      - Uses description + industry_tags if available
      - If company is in Stealth, infer from founder background
    Fallback: flat historical columns
    """

    startup_name = row.get("company_name") or "Unknown"
    about_text = row.get("about") if isinstance(row.get("about"), str) else ""

    row_text, prompt = "", ""

    if use_json:
        # Get current company info by index
        idx = row.get("company_index", 0)
        # Ensure idx is an integer
        try:
            idx = int(idx) if idx is not None else 0
        except (ValueError, TypeError):
            idx = 0
            
        all_experiences = row.get("all_experiences", [])
        
        # Safely get company info
        company_info = {}
        if isinstance(all_experiences, list) and len(all_experiences) > idx:
            company_info = all_experiences[idx] if isinstance(all_experiences[idx], dict) else {}
        
        company_name = company_info.get("company_name", "") or startup_name or ""
        description = company_info.get("description", "") or ""
        industry_tags = company_info.get("industry_tags", []) or []

        # Handle stealth case explicitly
        if company_name and "stealth" in company_name.lower():
            # Safely handle all_experiences - ensure it's a list/dict before json.dumps
            all_exp = row.get('all_experiences', [])
            if isinstance(all_exp, (list, dict)):
                exp_text = json.dumps(all_exp, indent=2)
            else:
                exp_text = str(all_exp) if all_exp else "No experience data"
            
            row_text = (
                f"Founder background and experiences: {exp_text}"
            )
            if about_text:
                row_text += f"\nMore about founder: {about_text}"

            prompt = (
                "This company is listed as Stealth. Based on the founder's professional background, roles, "
                "education, and any context provided, infer three verticals that the startup is most likely in. "
                "Respond ONLY with the three verticals and confidence levels.\n\n"
                "Examples:\n"
                "- AI (high), Healthcare (medium), Data Infrastructure (low)\n"
                "- SaaS (high), Cloud Security (medium), Analytics (medium)"
            )

        elif description or industry_tags:
            # Good case: we have description or tags
            row_text = f"Company: {company_name}"
            if description:
                row_text += f", Description: {description}"
            if industry_tags:
                row_text += f", Industry tags: {', '.join(industry_tags)}"
            if about_text:
                row_text += f"\nMore about founder: {about_text}"

            prompt = (
                "Based on the following description and industry tags of the founder's current company, "
                "identify three relevant verticals or industry terms the company likely operates in. "
                "Focus ONLY on the current company. Provide ONLY the verticals as comma-separated values "
                "with confidence levels.\n\n"
                "Examples:\n"
                "- Biotech (high), Genomics (medium), Health AI (medium)\n"
                "- SaaS (high), Enterprise Software (medium), Cloud Solutions (low)"
            )

        else:
            # Fallback → treat as no description/tags
            use_json = False  # force fallback to old style
            
        # Additional safety check - if we don't have meaningful data, fallback
        if use_json and not company_name and not description and not industry_tags:
            use_json = False

    if not use_json:
        # Old fallback mode (flat columns)
        relevant_columns = [
            "company_name_1", "position_1", "dates_1", "location_1", "description_1",
            "company_name_2", "position_2", "dates_2", "location_2", "description_2",
            "company_name_3", "position_3", "dates_3", "location_3", "description_3",
            "company_name_4", "position_4", "dates_4", "location_4", "description_4",
            "company_name_5", "position_5", "dates_5", "location_5", "description_5",
            "school_name_1", "degree_1", "school_dates_1", "details_1",
            "school_name_2", "degree_2", "school_dates_2", "details_2"
        ]
        row_text = ", ".join(
            [f"{col}: {row.get(col)}" for col in relevant_columns if row.get(col)]
        )
        if about_text:
            row_text += f"\nMore about founder: {about_text}"

        prompt = (
            f"This person is the founder of a company currently called {startup_name}. "
            "Based on their professional background, roles, and education, identify three verticals "
            "that likely describe what their current startup is doing. "
            "If the company is Stealth, guess from context. "
            "Provide ONLY the verticals, comma-separated, each with a confidence level.\n\n"
            "Examples:\n"
            "- Biotech (medium), Genomics (high), AI in Health (low)\n"
            "- SaaS (high), Cloud Computing (medium), Analytics (medium)\n"
            "- E-commerce (high), Consumer Finance (medium), Retail Tech (low)"
        )

    # Final GPT call
    response = ask_monty(prompt, row_text, max_tokens=150)
    return response.strip()

def get_past_notable_company(row, use_json=False):

    if use_json:
        row_text = ""
        all_experiences = row.get("all_experiences", [])
        if all_experiences and isinstance(all_experiences, list):
            for exp in all_experiences:
                if isinstance(exp, dict):
                    company_name = exp.get('company_name', '')
                    position = exp.get('position', '')
                    description = exp.get('description', '')
                    row_text += f"Company: {company_name}, Position: {position}, Description: {description}\n"
        
        # If no JSON data available, fall back to regular mode
        if not row_text.strip():
            use_json = False
    
    if not use_json: 
        relevant_columns = [
            "company_name_1", "position_1", "dates_1", "location_1", "description_1",
            "company_name_2", "position_2", "dates_2", "location_2", "description_2",
            "company_name_3", "position_3", "dates_3", "location_3", "description_3",
            "company_name_4", "position_4", "dates_4", "location_4", "description_4",
            "company_name_5", "position_5", "dates_5", "location_5", "description_5"
        ]
        row_text = ", ".join(
            [f"{column}: {value}" for column, value in row.items() if column in relevant_columns]
        )
        if row['about']:
            row_text += "If it helps, here is more information about the founder: " + row['about']

    prompt = (
        "Identify up to three impressive past employers from the founder's professional background. "
        "Impressive employers include Big Tech (e.g., FAANG), unicorns, Fortune 500 companies, and other prestigious firms. "
        "Exclude small startups and the founder’s current company. "
        "Be strict—only include truly noteworthy companies. "
        "If you have a JSON, use that. "
        "Output must follow this exact format: 'Tag1, Tag2' OR 'Tag1, Tag2, Tag3' OR 'Tag1'. "
        "No additional text, reasoning, or repetition. "
        "Use acronyms for well-known companies (e.g., IBM instead of International Business Machines)."
    )

    response = ask_monty(prompt, row_text, 150)

    return response


def get_past_notable_education(row):

    relevant_columns = [
        "school_name_1", "degree_1", "dates_1", "location_1", "description_1",
        "school_name_2", "degree_2", "dates_2", "location_2", "description_2",
        "school_name_3", "degree_3", "dates_3", "location_3", "description_3"
    ]
    row_text = ", ".join(
        [f"{column}: {value}" for column, value in row.items() if column in relevant_columns]
    )
    if row['about']:
        row_text += "If it helps, here is more information about the founder: " + row['about']

    prompt = (
        "Identify up to three impressive universities from the founder's educational background. "
        "Impressive universities include Ivy League schools, top global institutions (e.g., Stanford, Oxford, MIT), and other prestigious universities known for academic excellence. "
        "Exclude lesser-known institutions and online certifications. "
        "Be strict—only include truly noteworthy universities. "
        ""
        "Output must follow this exact format: 'Tag1, Tag2' OR 'Tag1, Tag2, Tag3' OR 'Tag1'. "
        "No additional text, reasoning, or repetition. "
        "Use acronyms where commonly used (e.g., MIT instead of Massachusetts Institute of Technology)."
    )

    response = ask_monty(prompt, row_text, 150)

    return response