from utils.profile_utils import construct_row_text
import re
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add the parent directory to the Python path so we can import the services module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.openai_api import ask_monty
from services.groq_api import get_groq_response

def parse_score(response):
    # Extracts the score and reasoning from the response
    score_match = re.search(r'\bScore: (\d+)\b', response)
    confidence_match = re.search(r'\bConfidence: (\d+)\b', response)
    reasoning_match = re.search(r'Reasoning: (.*)', response)

    score = int(score_match.group(1)) if score_match else 1  # Default to 1 if no score found
    confidence = int(confidence_match.group(1)) if confidence_match else 5  # Default confidence
    reasoning = reasoning_match.group(1) if reasoning_match else "No reasoning provided."

    return score, confidence, reasoning


def ai_score_founder(row):
    prompt = (
        "I am a venture capital investor at Montage Ventures looking for exceptional early-stage technology founders. "
        "I specialize in investing at the seed stage, so I am particularly interested in founders who are currently building "
        "at the pre-seed stage (less than 2 years old). "
        ""
        "The best fit for me is a founder who: "
        "- Has been building their company for less than 2 years (this is very important). "
        "- Has raised little institutional funding (Series A or further is NOT interesting, but pre-seed funding or grants is very interesting). "
        "- Has a strong technical background OR deep industry expertise. "
        "- Has prior experience at a top university, a well-known startup, or a prestigious tech company. "
        "- Has less than 10 years of total work experience. "
        "- Is based in the US. "
        ""
        "A low score should be given to founders who: "
        "- Are still in college or high school. "
        "- Have been building for over 2 years or have already raised Series A or later. "
        ""
        "Using the provided founder data, rate the fit of this founder on a scale from 1 to 10, where: "
        "1 = Definitely NOT a fit for Montage Ventures, "
        "10 = A perfect founder fit for our investment thesis. "
        ""
        "Also, provide a confidence level (1-10) and a **concise** reasoning (1-2 sentences). "
        "Strictly structure the response as: "
        ""
        "Score: \n"
        "Confidence: \n"
        "Reasoning: \n"
    )

    row_text = construct_row_text(row)
    if row['about'] and row['about'] != "":
        row_text += "If it helps, here is more information about the founder: " + row['about']
    response = ask_monty(prompt, row_text, 100)
    score, confidence, reasoning = parse_score(response)
    return score

def ai_score_fintech(row):
    prompt = (
        "I am an investor at Montage Ventures looking to invest in early-stage fintech companies. "
        "I am specifically interested in founders who are building in any of the following fintech themes:\n"
        "\n"
        "1. Fighting financial fraud:\n"
        "   - cybersecurity\n"
        "   - transaction monitoring\n"
        "   - account provisioning\n"
        "   - identity gathering for onboarding\n"
        "\n"
        "2. Transforming cross-border payments in Latin America:\n"
        "   - B2B cross-border transactions\n"
        "   - Remittance solutions\n"
        "   - Blockchain & crypto\n"
        "\n"
        "3. AI for financial authentication:\n"
        "   - Advanced biometric solutions\n"
        "   - AI-powered document verification\n"
        "   - Behavioral biometrics\n"
        "   - Decentralized identity solutions\n"
        "   - Multi-factor authentication platforms\n"
        "\n"
        "4. Transforming finance through agentic AI:\n"
        "   - Autonomous trading systems\n"
        "   - Intelligent risk management\n"
        "   - Personalized financial advisors\n"
        "   - Regulatory compliance agents\n"
        "   - Dynamic credit scoring\n"
        "\n"
        "5. Modernizing the insurance industry AI:\n"
        "   - New insurance products for AI\n"
        "   - Addressing climate risk\n"
        "   - AI powered client interactions\n"
        "   - Intelligent underwriting and decisioning\n"
        "   - Claims automation\n"
        "   - New data models and providers\n"
        "\n"
        "6. The great wealth shift:\n"
        "   - AI and automation in wealth management\n"
        "   - Embedded finance and wealth services\n"
        "   - Real estate inheritance tech and liquidity solutions\n"
        "   - Digital identity and security for wealth transfers\n"
        "   - Multi-generational financial ecosystems\n"
        "\n"
        "I am looking for founders who:\n"
        "- Started their company within the past 1-2 years\n"
        "- Are based in the US\n"
        "- Have technical expertise or deep domain knowledge in financial services\n"
        "- Are building innovative solutions in one of the themes above\n"
        "\n"
        "Based on this person's background, product description, target market, and technical uniqueness score, "
        "give me a score (1-10), a confidence level (1-10), and a short reasoning behind that score.\n"
        "\n"
        "Pay special attention to:\n"
        "- The product description and what problem it solves\n"
        "- The target market and who they're serving\n"
        "- The technical uniqueness score as an indicator of innovation\n"
        "\n"
        "A score of 1 would indicate that:\n"
        "- This is definitely NOT someone building in fintech, OR\n"
        "- They're building something that doesn't align with any of my themes of interest\n"
        "\n"
        "A score of 10 would indicate that:\n"
        "- This person is definitely building an early-stage fintech company\n"
        "- Their company aligns perfectly with one of my themes of interest\n"
        "- They have the right background and are at the ideal stage for Montage Ventures\n"
        "\n"
        "If the person seems very knowledgeable in topics that relate to fintech like AI, but their company "
        "is not in fintech, they get a lower score. I am only looking for founders in fintech, preferably "
        "early in their journey!\n"
        "\n"
        "It's very important that you structure the response as:\n"
        "Score: \n"
        "Confidence: \n"
        "Reasoning: \n"
    )

    row_text = construct_row_text(row)
    if row['about'] and row['about'] != "":
        row_text += "If it helps, here is more information about the founder: " + row['about']
    response = ask_monty(prompt, row_text, 100)
    score, confidence, reasoning = parse_score(response)
    return score


def ai_score_healthcare(row):
    prompt = (
        "I am an investor at Montage Ventures focused on investing in healthcare and healthtech. "
        "I am specifically interested in founders who are building in the following healthcare themes:\n"
        "\n"
        "1. AI for drug discovery:\n"
        "   - Bespoke data advantage\n"
        "   - Multiomics\n"
        "   - Techbio\n"
        "\n"
        "2. Revolutionizing oncology care with AI-driven innovation:\n"
        "   - AI-powered diagnostics\n"
        "   - Drug discovery platforms\n"
        "   - Clinical trials optimization\n"
        "   - Personalized treatment planning\n"
        "   - AI coordination and navigation\n"
        "\n"
        "3. Providers of the future:\n"
        "   - concierge care\n"
        "   - specialist care\n"
        "   - longevity\n"
        "\n"
        "I am looking for founders who:\n"
        "- Started their company within the past 1-2 years\n"
        "- Are based in the US\n"
        "- Have technical expertise, clinical background, or deep domain knowledge in healthcare\n"
        "- Are building innovative solutions in one of the themes above\n"
        "\n"
        "Based on this person's background, product description, target market, and technical uniqueness score, "
        "give me a score (1-10), a confidence level (1-10), and a short reasoning behind that score.\n"
        "\n"
        "Pay special attention to:\n"
        "- The product description and what healthcare problem it solves\n"
        "- The target market and which healthcare segments they're serving\n"
        "- The technical uniqueness score as an indicator of healthcare innovation\n"
        "\n"
        "A score of 1 would indicate that:\n"
        "- This is definitely not someone building in healthcare/healthtech, OR\n"
        "- They're building something that doesn't align with any of my themes of interest\n"
        "\n"
        "A score of 10 would indicate that:\n"
        "- This person is definitely building an early-stage healthcare/healthtech company\n"
        "- Their company aligns perfectly with one of my themes of interest\n"
        "- They have the right background and are at the ideal stage for Montage Ventures\n"
        "\n"
        "It's important that you structure the response as:\n"
        "Score: \n"
        "Confidence: \n"
        "Reasoning: \n"
    )

    row_text = construct_row_text(row)
    if row['about'] and row['about'] != "":
        row_text += "If it helps, here is more information about the founder: " + row['about']
    response = ask_monty(prompt, row_text, 100)
    score, confidence, reasoning = parse_score(response)

    return score


def ai_score_commerce(row):
    prompt = (
        "I am an investor at Montage Ventures looking to invest in early-stage commerce and retail tech companies. "
        "I am specifically interested in founders who are building in the following commerce themes:\n"
        "\n"
        "1. AI-powered commerce tools:\n"
        "   - Generative AI for product discovery\n"
        "   - AI-driven inventory management\n"
        "   - Personalized shopping experiences\n"
        "   - Visual search and recognition\n"
        "\n"
        "2. Next-generation retail infrastructure:\n"
        "   - Omnichannel solutions\n"
        "   - Modern POS systems\n"
        "   - Retail analytics platforms\n"
        "   - In-store technology\n"
        "\n"
        "3. Supply chain and logistics optimization:\n"
        "   - Last-mile delivery solutions\n"
        "   - Warehouse automation\n"
        "   - Sustainable logistics\n"
        "   - Inventory forecasting\n"
        "\n"
        "4. Commerce enablement for brands and retailers:\n"
        "   - DTC enablement tools\n"
        "   - Marketplace solutions\n"
        "   - Brand management platforms\n"
        "   - Customer acquisition tools\n"
        "\n"
        "5. Innovative shopping experiences:\n"
        "   - Social commerce\n"
        "   - AR/VR shopping\n"
        "   - Live shopping platforms\n"
        "   - Community-driven commerce\n"
        "\n"
        "I am looking for founders who:\n"
        "- Started their company within the past 1-2 years\n"
        "- Are based in the US\n"
        "- Have technical expertise or deep domain knowledge in commerce/retail\n"
        "- Are building innovative solutions in one of the themes above\n"
        "\n"
        "Based on this person's background, product description, target market, and technical uniqueness score, "
        "give me a score (1-10), a confidence level (1-10), and a short reasoning behind that score.\n"
        "\n"
        "Pay special attention to:\n"
        "- The product description and what commerce/retail problem it solves\n"
        "- The target market and which retail segments or consumers they're serving\n"
        "- The technical uniqueness score as an indicator of retail innovation\n"
        "\n"
        "A score of 1 would indicate that:\n"
        "- This is definitely NOT someone building in commerce/retail tech, OR\n"
        "- They're building something that doesn't align with any of my themes of interest\n"
        "\n"
        "A score of 10 would indicate that:\n"
        "- This person is definitely building an early-stage commerce/retail tech company\n"
        "- Their company aligns perfectly with one of my themes of interest\n"
        "- They have the right background and are at the ideal stage for Montage Ventures\n"
        "\n"
        "It's important that you structure the response as:\n"
        "Score: \n"
        "Confidence: \n"
        "Reasoning: \n"
    )

    row_text = construct_row_text(row)
    if row['about'] and row['about'] != "":
        row_text += "If it helps, here is more information about the founder: " + row['about']
    response = ask_monty(prompt, row_text, 100)
    score, confidence, reasoning = parse_score(response)

    return score

def prompt_industry_score(row, json=False):

    prompt = (
        "You are an intelligent analyst at Montage Ventures. "
        "Consider the background of the following profile. Return a score between 0 and 10 "
        "that accurately reflects the industry experience of this person. "
        "A score of 0 means the person is still in school with no work experience. "
        "A score of 1 means they are still in school but have some internship experiences. "
        "A score of 2 means they just graduated but have less than one year of work experience. "
        "A score between 3 and 8 should reflect their cumulative years in the industry, "
        "the quality of the companies they worked at, and their seniority in roles. "
        "A score of 10 means they have extremely deep expertise in their industry. "
        "Based on this, provide a single number as the score - no additional text."
    )

    # Build profile data
    data = ""
    if json:
        experiences = row.get("all_experiences", [])
        for exp in experiences:
            data += f"(Company: {exp.get('company_name', 'Unknown')}, Position: {exp.get('position', 'Unknown')}, Description: {exp.get('description', 'Unknown')}) "
    else:
        for i in range(1, 6):
            company = f"company_name_{i}"
            position = f"position_{i}"
        description = f"description_{i}"

        # Ensure values exist to avoid errors
        company_value = row.get(company, "Unknown")
        position_value = row.get(position, "Unknown")
        dates_value = row.get(dates, "Unknown")
        description_value = row.get(description, "Unknown")

        data += f"(Company: {company_value}, Position: {position_value}, Dates: {dates_value}, Description: {description_value}) "

    # Add years of experience
    years_experience = str(row.get('years_of_experience', "Unknown"))
    about_text = row.get('about', "No LinkedIn bio available.")

    data += f"They graduated their highest education around {years_experience} years ago. "
    data += f"This is their LinkedIn bio: {about_text}"

    # Query AI model
    response = ask_monty(prompt, data, 50)

    # Ensure response is a valid integer
    try:
        score = int(response.strip())  # Strip extra spaces and convert to int
    except ValueError:
        print(f"Warning: AI returned an invalid response: {response}")
        score = -1  # Default or error value if parsing fails

    return score


def startup_experience_score(row, json=False):
    prompt = (
        "I am a venture capital investor evaluating a potential founder's startup experience. "
        "I need to assess their experience founding, working at, or advising startups on a scale from 0 to 10. "
        ""
        "Please evaluate this person's startup experience based on the following criteria: "
        ""
        "0-2: Limited or no startup experience (e.g., only worked at large companies or is a recent graduate) "
        "3-4: Some exposure to startups (e.g., worked at a late-stage startup, minor advisory roles) "
        "5-6: Moderate startup experience (e.g., early employee at a successful startup, founded a small startup) "
        "7-8: Significant startup experience (e.g., founded a venture-backed startup, early employee at a unicorn) "
        "9-10: Exceptional startup experience (e.g., multiple successful exits as a founder, key executive at a unicorn) "
        ""
        "Consider factors such as: "
        "- Whether they've founded companies before "
        "- Their roles at startups (founder, early employee, executive) "
        "- The success of startups they've been involved with "
        "- The relevance of their startup experience to their current venture "
        ""
        "Please provide a score (0-10), a confidence level (1-10), and a brief reasoning. "
        "Structure your response exactly as: "
        ""
        "Score: "
        "Confidence: "
        "Reasoning: "
    )

    row_text = construct_row_text(row, json=json)
    if row['about'] and row['about'] != "" and str(row['about']) != 'nan':
        row_text += "If it helps, here is more information about the founder: " + str(row['about'])
    response = get_groq_response(prompt + "\n\nPerson's profile:\n" + row_text)
    score, confidence, reasoning = parse_score(response)
    return score

def past_success_indication_score(row, json=False):
    prompt = (
        "I am a venture capital investor evaluating a potential founder's track record of success. "
        "I need to assess their past achievements and indicators of future success on a scale from 0 to 10. "
        ""
        "Please evaluate this person's track record based on the following criteria: "
        ""
        "0-2: Limited or no indicators of past success (e.g., no notable education, no prestigious employers) "
        "3-4: Some indicators of success (e.g., decent university, some career progression) "
        "5-6: Moderate indicators of success (e.g., good university, solid career progression) "
        "7-8: Strong indicators of success (e.g., top university, impressive career trajectory, leadership roles) "
        "9-10: Exceptional indicators of success (e.g., elite university, rapid career advancement, significant achievements) "
        ""
        "Consider factors such as: "
        "- Educational background (quality of institutions, degrees obtained) "
        "- Career progression and growth "
        "- Leadership positions held "
        "- Notable achievements or recognition "
        "- Demonstrated ability to overcome challenges "
        ""
        "Please provide a score (0-10), a confidence level (1-10), and a brief reasoning. "
        "Structure your response exactly as: "
        ""
        "Score: "
        "Confidence: "
        "Reasoning: "
    )

    row_text = construct_row_text(row, json=json)
    if row['about'] and row['about'] != "" and str(row['about']) != 'nan':
        row_text += "If it helps, here is more information about the founder: " + str(row['about'])
    response = get_groq_response(prompt + "\n\nPerson's profile:\n" + row_text)
    score, confidence, reasoning = parse_score(response)
    return score

def company_tech_score(row, json=False):
    """
    Evaluates how technically advanced and unique a company's product/service is on a scale of 0-10.
    
    Args:
        row (dict): Dictionary containing company and product information
        
    Returns:
        int: Score from 0-10 indicating technical uniqueness
    """
    prompt = (
        "I am a venture capital investor evaluating the technical uniqueness and innovation level of a startup. "
        "I need to assess how technically advanced and unique their product or service is on a scale from 0 to 10. "
        ""
        "Please evaluate this company's technical uniqueness based on the following criteria: "
        ""
        "0-2: Very low/non-technical, non-unique solution. There are better technologies already in the market. "
        "     - No technical innovation required to build this product/service "
        "     - Purely repackaging existing solutions "
        "     - No barriers to entry for competitors "
        ""
        "3-4: Low technical uniqueness. Could be built by other teams with basic CS/engineering backgrounds. "
        "     - Minimal technical innovation "
        "     - Primarily using off-the-shelf technologies "
        "     - Low technical barriers to entry "
        ""
        "5-6: Modest technical uniqueness. Not a significant breakthrough but some improvements to existing products. "
        "     - Some technical innovation "
        "     - Combines existing technologies in novel ways "
        "     - Moderate technical barriers to entry "
        ""
        "7-8: Highly technical. Significant improvements to existing technologies using years of experience and domain knowledge. "
        "     - Substantial technical innovation "
        "     - Requires deep expertise in specific technical domains "
        "     - High technical barriers to entry "
        ""
        "9-10: Extremely unique and technically advanced. Completely changes how things currently work, requires deep domain expertise. "
        "      - Revolutionary technical innovation "
        "      - Creates entirely new technical approaches or paradigms "
        "      - Very high technical barriers to entry "
        "      - Often based on proprietary research or breakthroughs "
        ""
        "Pay special attention to: "
        "- Technical complexity of the solution "
        "- Uniqueness compared to existing solutions "
        "- Technical barriers to entry for competitors "
        "- Required expertise to build the solution "
        "- Presence of proprietary technology or IP "
        "- Use of cutting-edge technologies (AI, ML, blockchain, etc.) in novel ways "
        ""
        "Using the provided company information, rate this company's technical uniqueness on a scale from 0 to 10. "
        "Also, provide a confidence level (1-10) and a **concise** reasoning (1-2 sentences). "
        "Strictly structure the response as: "
        ""
        "Score: "
        "Confidence: "
        "Reasoning: "
    )

    # Construct text about the company from available information
    company_text = ""
    
    # Add product description if available
    if row.get('product_description') or row.get('product'):
        company_text += "Product description: " + (row.get('product_description') or row.get('product', '')) + "\n\n"
    
    # Add market description if available
    if row.get('market_description'):
        company_text += "Target market: " + row.get('market_description', '') + "\n\n"
    
    # Add company description from LinkedIn if available
    if row.get('description_1'):
        company_text += "Company description: " + row.get('description_1', '') + "\n\n"
    
    # Add company name and founder's title if available
    if row.get('company_name'):
        company_text += "Company name: " + row.get('company_name', '') + "\n\n"
    
    if row.get('title'):
        company_text += "Founder's title: " + row.get('title', '') + "\n\n"
    
    # Add verticals if available
    if row.get('verticals'):
        company_text += "Business verticals: " + row.get('verticals', '') + "\n\n"
    
    # If we have very little information, return a default score
    if len(company_text.strip()) < 50:
        return 5  # Default middle score when insufficient information
    
    response = get_groq_response(prompt + "\n\nCompany information:\n" + company_text)
    score, confidence, reasoning = parse_score(response)
    return score
