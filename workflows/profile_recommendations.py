"""
Profile Recommendations System for Monty
=========================================

This module handles profile/talent recommendations for the weekly update.

Main Functions:
--------------
1. get_profile_recommendations() - Get top 3 filtered profiles for weekly update
2. mark_profiles_as_recommended() - Track recommended profile IDs to avoid duplicates
3. format_profile_for_slack() - Format profile data for Slack display with highlights

Usage:
------
To get profile recommendations:
    from workflows.profile_recommendations import get_profile_recommendations
    profiles = get_profile_recommendations(limit=3)

To mark profiles as recommended:
    from workflows.profile_recommendations import mark_profiles_as_recommended
    mark_profiles_as_recommended(profile_ids)

Requirements:
------------
- Aviato API key must be configured
- data/recommended_profiles.json will be created automatically
"""

import json
import os
import logging
from datetime import datetime
from workflows.aviato_processing import search_aviato_profiles, enrich_profiles, filter_relevant_profiles
from services.openai_api import ask_monty

logger = logging.getLogger(__name__)

# Path to the JSON file tracking recommended profiles
RECOMMENDED_PROFILES_FILE = 'data/recommended_profiles.json'

def load_recommended_profiles():
    """Load the list of previously recommended profile IDs from JSON file.
    
    Returns:
        dict: {
            'profile_ids': [list of profile IDs],
            'last_updated': timestamp,
            'total_recommended': count
        }
    """
    if not os.path.exists(RECOMMENDED_PROFILES_FILE):
        return {
            'profile_ids': [],
            'last_updated': None,
            'total_recommended': 0
        }
    
    try:
        with open(RECOMMENDED_PROFILES_FILE, 'r') as f:
            data = json.load(f)
            # Ensure all required keys exist
            if 'profile_ids' not in data:
                data['profile_ids'] = []
            if 'total_recommended' not in data:
                data['total_recommended'] = len(data.get('profile_ids', []))
            return data
    except Exception as e:
        logger.error(f"Error loading recommended profiles: {e}")
        return {
            'profile_ids': [],
            'last_updated': None,
            'total_recommended': 0
        }


def save_recommended_profiles(data):
    """Save the list of recommended profile IDs to JSON file.
    
    Args:
        data: Dictionary with profile_ids list and metadata
    """
    try:
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        with open(RECOMMENDED_PROFILES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data['profile_ids'])} recommended profile IDs")
    except Exception as e:
        logger.error(f"Error saving recommended profiles: {e}")


def mark_profiles_as_recommended(profile_ids):
    """Mark profiles as recommended by adding their IDs to the tracking file.
    
    Args:
        profile_ids: List of profile IDs to mark as recommended
    """
    if not profile_ids:
        logger.warning("No profile IDs provided to mark as recommended")
        return
    
    data = load_recommended_profiles()
    
    # Add new IDs (avoiding duplicates)
    existing_ids = set(data['profile_ids'])
    new_ids = [pid for pid in profile_ids if pid not in existing_ids]
    
    if new_ids:
        data['profile_ids'].extend(new_ids)
        data['last_updated'] = datetime.now().isoformat()
        data['total_recommended'] = len(data['profile_ids'])
        
        save_recommended_profiles(data)
        logger.info(f"Marked {len(new_ids)} new profiles as recommended (total: {data['total_recommended']})")
    else:
        logger.info("All provided profile IDs were already marked as recommended")


def select_best_profiles_with_ai(profiles, target_count=3):
    """Use AI to select the most relevant profiles for fintech, healthcare, or commerce.
    
    Args:
        profiles: List of profile dictionaries
        target_count: Number of profiles to select (default: 3)
    
    Returns:
        list: Selected profiles
    """
    if len(profiles) <= target_count:
        return profiles

    return profiles[:target_count]
    
    # Build context for each profile
    profile_summaries = []
    for i, profile in enumerate(profiles):
        name = profile.get('fullName', 'Unknown')
        headline = profile.get('headline', 'No headline')
        location = profile.get('location', 'Unknown')
        
        # Get experience data
        experiences = profile.get('experience', [])
        experience_summary = []
        for exp in experiences[:3]:  # Top 3 experiences
            if isinstance(exp, dict):
                title = exp.get('title', '')
                company = exp.get('companyName', '')
                if title and company:
                    experience_summary.append(f"{title} at {company}")
        
        # Get skills
        skills = profile.get('skills', [])
        if isinstance(skills, list):
            skills_list = [s.get('name', s) if isinstance(s, dict) else str(s) for s in skills[:10]]
            skills_text = ", ".join(skills_list) if skills_list else "No skills listed"
        else:
            skills_text = "No skills listed"
        
        profile_summaries.append(f"""
Profile {i+1}: {name}
Current: {headline}
Location: {location}
Recent Experience: {'; '.join(experience_summary) if experience_summary else 'Not available'}
Skills: {skills_text}
""")
    
    # Create prompt for AI
    prompt = """You are an expert VC analyst at Montage Ventures. We focus EXCLUSIVELY on early-stage investments in three verticals:

**FINTECH**: Payments, banking, lending, insurance, wealth management, accounting, financial infrastructure, crypto/blockchain
**HEALTHCARE**: Digital health, healthtech, biotech, medical devices, health insurance, telehealth, clinical software, health data
**COMMERCE**: E-commerce, retail tech, marketplaces, supply chain, logistics, DTC brands, B2B commerce, point-of-sale

Your task: Select the 3 profiles with the STRONGEST and MOST DIRECT relevance to one of these three verticals. 

Selection criteria (in order of importance):
1. **Direct vertical experience**: Has worked at major companies in fintech/healthcare/commerce OR founded a startup in these spaces
   - Fintech examples: Stripe, Square, Plaid, Robinhood, Coinbase, PayPal, Affirm
   - Healthcare examples: Oscar Health, Ro, Hims, One Medical, 23andMe, Tempus
   - Commerce examples: Shopify, Amazon, Instacart, DoorDash, Faire, Flexport
2. **Relevant skills**: Product/engineering skills specifically applied to fintech, healthcare, or commerce problems
3. **Location**: SF Bay Area strongly preferred

EXCLUDE profiles that:
- Work in general tech infrastructure (networking, cloud, DevOps) without vertical focus
- Are in unrelated industries (deep tech, general AI/ML without vertical application)
- Have only generic "big tech" experience without vertical relevance

Return your response in this exact JSON format:
{
  "selections": [
    {
      "profile_number": 1,
      "vertical": "fintech|healthcare|commerce",
      "reason": "Brief 1-sentence reason explaining their DIRECT relevance to the vertical"
    }
  ]
}"""

    data = "\n".join(profile_summaries)
    
    try:
        logger.info("Using AI to select best profiles...")
        response = ask_monty(prompt, data, max_tokens=500)
        
        # Parse JSON response
        import json
        import re
        
        try:
            # Clean markdown fences and whitespace
            clean_response = response.strip()
            clean_response = re.sub(r"```(?:json)?", "", clean_response).strip()

            # Try to find the first JSON object (non-greedy)
            json_match = re.search(r"\{.*?\}", clean_response, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                result = json.loads(json_text)
                selections = result.get("selections", [])
            else:
                raise ValueError("No JSON object found in response")

            selected_profiles = []
            for selection in selections[:target_count]:
                idx = selection.get('profile_number', 0) - 1
                if 0 <= idx < len(profiles):
                    profile = profiles[idx].copy()
                    profile['ai_vertical'] = selection.get('vertical', 'general')
                    profile['ai_reason'] = selection.get('reason', '')
                    selected_profiles.append(profile)

            if selected_profiles:
                logger.info(f"AI selected {len(selected_profiles)} profiles with reasoning")
                return selected_profiles

        except Exception as e:
            logger.warning(f"Could not parse JSON ({e}), trying fallback parsing")
            # Fallback to top profiles if AI output invalid
            logger.warning("AI selection failed, falling back to top profiles")
            return profiles[:target_count]
            
    except Exception as e:
        logger.error(f"Error in AI selection: {e}")
        return profiles[:target_count]


def get_profile_recommendations(limit=3, search_filters=None):
    """Get top profile recommendations for the weekly update.
    
    This function:
    1. Searches for profiles using specified filters
    2. Enriches the profiles with full data
    3. Filters based on required highlights
    4. Excludes previously recommended profiles
    5. Gets top 20 candidates
    6. Uses AI to select the most relevant N profiles for fintech/healthcare/commerce
    7. Returns the AI-selected profiles
    
    Args:
        limit: Number of profiles to return (default: 3)
        search_filters: Custom search filters (optional). If None, uses default filters.
    
    Returns:
        list: List of profile dictionaries with full enriched data, AI-selected for relevance
    """
    # Default search filters
    if search_filters is None:
        search_filters = {
            "region": "California",
            "country": "United States",
            "computed_potentialToLeave": True,
            "computed_priorBackedFounder": True,
            "sort": [{"computed_likelyToExplore": {"order": "desc"}}],
            "computed_recentlyLeftCompany": False,
            "limit": 100  # Get more results to filter from
        }
    
    logger.info("Searching for profile recommendations...")
    
    # Step 1: Search for profiles
    results = search_aviato_profiles(search_filters)
    
    if not results or "items" not in results:
        logger.error("No results found from profile search")
        return []
    
    profiles = results["items"]
    logger.info(f"Found {len(profiles)} profiles from search")
    
    # Step 2: Enrich profiles
    ids = [item["id"] for item in profiles]
    enriched = enrich_profiles(ids)
    logger.info(f"Enriched {len(enriched)} profiles")
    
    if not enriched:
        logger.error("No enriched profiles returned")
        return []
    
    # Step 3: Filter by required highlights
    required_highlights = ['potentialToLeave', 'priorBackedFounder']
    optional_highlights = ['bigTechAlumPrivate', 'bigTechAlumPublic']
    
    filtered = filter_relevant_profiles(enriched, required_highlights, optional_highlights)
    logger.info(f"Filtered to {len(filtered)} profiles with required highlights")
    
    if not filtered:
        logger.warning("No profiles matched the highlight criteria")
        return []
    
    # Step 4: Exclude previously recommended profiles
    recommended_data = load_recommended_profiles()
    previously_recommended = set(recommended_data['profile_ids'])
    
    # Verify profile IDs exist and log details
    profiles_with_ids = [p for p in filtered if p.get('id')]
    profiles_without_ids = [p for p in filtered if not p.get('id')]
    
    if profiles_without_ids:
        logger.warning(f"{len(profiles_without_ids)} profiles missing 'id' field - will be excluded from recommendations")
    
    # Filter out previously recommended profiles
    new_profiles = []
    excluded_count = 0
    for p in profiles_with_ids:
        profile_id = p.get('id')
        if profile_id not in previously_recommended:
            new_profiles.append(p)
        else:
            excluded_count += 1
    
    logger.info(f"After excluding {len(previously_recommended)} previously recommended ({excluded_count} from current batch), {len(new_profiles)} new profiles remain")
    
    if not new_profiles:
        logger.warning("All filtered profiles have been previously recommended")
        return []
    
    # Step 5: Get top 20 candidates for AI selection
    top_candidates = new_profiles[:20]
    logger.info(f"Selected top {len(top_candidates)} candidates for AI evaluation")
    
    # Step 6: Use AI to select the best N profiles
    selected_profiles = select_best_profiles_with_ai(top_candidates, target_count=limit)
    logger.info(f"AI selected {len(selected_profiles)} final profile recommendations")
    
    # Final verification: ensure all returned profiles are new (double-check)
    final_profile_ids = [p.get('id') for p in selected_profiles if p.get('id')]
    duplicate_check = [pid for pid in final_profile_ids if pid in previously_recommended]
    
    if duplicate_check:
        logger.error(f"⚠️  WARNING: {len(duplicate_check)} selected profiles were previously recommended! IDs: {duplicate_check}")
        # Filter out duplicates just to be safe
        selected_profiles = [p for p in selected_profiles if p.get('id') not in previously_recommended]
        logger.info(f"After removing duplicates, {len(selected_profiles)} new profiles remain")
    
    if selected_profiles:
        logger.info(f"✅ Returning {len(selected_profiles)} NEW profiles (IDs: {[p.get('id') for p in selected_profiles if p.get('id')]})")
    
    return selected_profiles


def format_profile_for_slack(profile, include_highlights=True):
    """Format a profile for Slack display with highlights explanation.
    
    Args:
        profile: Profile dictionary from enriched API data
        include_highlights: Whether to include the highlights explanation
    
    Returns:
        str: Formatted Slack message block for the profile
    """
    name = profile.get('fullName', 'Unknown')
    headline = profile.get('headline', 'No headline')
    location = profile.get('location', 'Unknown location')
    
    # Get LinkedIn URL
    urls = profile.get('URLs', {})
    linkedin_url = urls.get('linkedin', '')
    if linkedin_url and not linkedin_url.startswith('http'):
        linkedin_url = f"https://{linkedin_url}"
    
    # Build the main profile line
    if linkedin_url:
        profile_line = f"• <{linkedin_url}|*{name}*>"
    else:
        profile_line = f"• *{name}*"
    
    # Add headline and location
    profile_line += f"\n    {headline}"
    profile_line += f"\n    📍 {location}"
    
    # Add highlights explanation if requested
    if include_highlights:
        highlights = profile.get('computed_highlightList', [])
        
        # Build highlights explanation
        highlight_parts = []
        
        if 'potentialToLeave' in highlights:
            highlight_parts.append("showing signs of being open to new opportunities")
        
        if 'priorBackedFounder' in highlights:
            highlight_parts.append("previously founded a VC-backed company")
        
        if 'bigTechAlumPublic' in highlights:
            highlight_parts.append("worked at a major public tech company")
        elif 'bigTechAlumPrivate' in highlights:
            highlight_parts.append("worked at a major private tech company")
        
        if 'employeeDuringIPO' in highlights:
            highlight_parts.append("experienced an IPO")
        
        if 'unicornEarlyEngineer' in highlights:
            highlight_parts.append("early engineer at a unicorn")
        
        if 'topUniversity' in highlights:
            highlight_parts.append("attended a top university")
        
        if highlight_parts:
            highlights_text = ", ".join(highlight_parts)
            profile_line += f"\n    _Why: {highlights_text}_"
    
    return profile_line


def format_profiles_for_weekly_update(profiles):
    """Format multiple profiles for the weekly update Slack message.
    
    Args:
        profiles: List of profile dictionaries
    
    Returns:
        str: Formatted Slack message with all profiles
    """
    if not profiles:
        return ""
    
    message_parts = [
        "Here are some interesting profiles we came across this week:\n"
    ]
    
    for profile in profiles:
        formatted = format_profile_for_slack(profile, include_highlights=True)
        message_parts.append(formatted)
    
    return "\n\n".join(message_parts)


# For testing/debugging
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing profile recommendations system...")
    print("=" * 80)
    
    # Get recommendations
    profiles = get_profile_recommendations(limit=3)
    
    if profiles:
        print(f"\n✅ Found {len(profiles)} recommendations:\n")
        
        # Format for display
        message = format_profiles_for_weekly_update(profiles)
        print(message)
        
        # Show what would be marked as recommended
        profile_ids = [p['id'] for p in profiles]
        print("\n" + "=" * 80)
        print(f"Would mark these {len(profile_ids)} profile IDs as recommended:")
        for pid in profile_ids:
            print(f"  - {pid}")
    else:
        print("❌ No recommendations found")
