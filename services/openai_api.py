from openai import OpenAI
import anthropic
import os
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv()

# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")

def ask_monty(prompt, data, max_tokens=1000, model=None):
    """
    Call Claude chat completion.

    Args:
        prompt: System message / instructions.
        data: User message / input data.
        max_tokens: Maximum tokens in the response.
        model: Model name. If None, uses claude-haiku-4-5-20251001.
    """
    if model is None:
        model = "claude-haiku-4-5-20251001"
    client = anthropic.Anthropic(api_key=claude_api_key)
    response = client.messages.create(
        model=model,
        system=prompt,
        messages=[{"role": "user", "content": data}],
        max_tokens=max_tokens,
    )
    return response.content[0].text


def generate_talent_description(all_experiences, person_name, max_tokens: int = 150) -> str:
    """
    Generate a personalized description for a talent recommendation based on their experience.
    
    Analyzes all_experiences to identify the company they founded and their background,
    then generates a short 1-2 sentence description about what type of company they built
    and their relevant experience.
    
    Args:
        all_experiences: List of experience dictionaries, each with:
            - company_name: str
            - position: str
            - start_date: str (optional)
            - end_date: str (optional, None if current)
            - description: str (optional)
        person_name: Name of the person (for context)
        max_tokens: Max tokens for the model output.
    
    Returns:
        A short string (1-2 sentences, max ~60 words) describing their background and company.
    """
    import json
    
    # Guard rails
    if not all_experiences or not isinstance(all_experiences, list) or len(all_experiences) == 0:
        return None
    
    # Format experiences for the prompt
    experiences_text = []
    for exp in all_experiences:
        company = exp.get("company_name", "Unknown Company")
        position = exp.get("position", "")
        start_date = exp.get("start_date", "")
        end_date = exp.get("end_date")
        description = exp.get("description", "")
        
        # Build experience line
        exp_line = f"- {position} at {company}"
        if start_date:
            exp_line += f" ({start_date}"
            if end_date:
                exp_line += f" - {end_date})"
            else:
                exp_line += " - present)"
        if description:
            exp_line += f": {description[:200]}"  # Truncate long descriptions
        
        experiences_text.append(exp_line)
    
    experiences_str = "\n".join(experiences_text)
    
    prompt = (
        "You are a concise VC investment analyst at Montage Ventures writing a weekly newsletter. "
        "Analyze the person's work experience below and generate a short, straightforward description (1-2 sentences, max ~60 words) "
        "that describes their background and why they might be an interesting founder candidate.\n\n"
        "REQUIREMENTS:\n"
        "1. If they are a founder: Describe what type of company they founded/built (be specific: 'a B2B payments API', 'a telehealth platform for mental health', 'a DTC skincare brand')\n"
        "2. Describe their prior experience at specific companies (e.g., 'previously led product at Stripe', 'ex-Google engineer', 'worked on payments infrastructure at Square')\n"
        "3. Mention relevant education if notable (e.g., 'Stanford CS graduate', 'MIT MBA')\n\n"
        "CRITICAL - AVOID THESE GENERIC PHRASES:\n"
        "- 'showcasing expertise in...'\n"
        "- 'brings a wealth of...'\n"
        "- 'demonstrates strong...'\n"
        "- 'innovative tech solutions'\n"
        "- 'technical expertise and leadership skills'\n"
        "- 'proven track record' (unless you can be specific)\n"
        "- Any vague or marketing-style language\n\n"
        "INSTEAD, be direct and factual:\n"
        "- 'Founded [Company], a [specific type of business] that [what it does]'\n"
        "- 'Previously [specific role] at [Company] where [specific achievement/area]'\n"
        "- 'Built [specific product/feature] at [Company]'\n"
        "- 'Serial entrepreneur with exits in [specific industries]'\n\n"
        "Write in a natural, straightforward tone. Focus on concrete facts about their ventures, work history, and education. "
        "Do not include headings, bullet points, or the person's name in the description."
    )
    
    data = f"Person: {person_name}\n\nWork Experience:\n{experiences_str}"
    
    try:
        description = ask_monty(prompt, data, max_tokens=max_tokens)
        return (description or "").strip()
    except Exception as e:
        print(f"Error generating talent description: {e}")
        return None


def summarize_company_recommendation(interest: str,
                                     company_description: str,
                                     company_name: str = None,
                                     subcategory: str = None,
                                     max_tokens: int = 140) -> str:
    """
    Produce a concise 1–2 sentence newsletter-ready blurb explaining why we're recommending this company
    by tying the subcategory "interest" notes to the company's description.

    Args:
        interest: Investment interest text pulled from the taste tree (subcategory meta.interest).
        company_description: The company's description (product/what they do).
        company_name: Optional company name for context.
        subcategory: Optional subcategory name for context.
        max_tokens: Max tokens for the model output.

    Returns:
        A short string (1–2 sentences) suitable for inclusion in the newsletter.
    """
    # Guard rails and sane defaults
    interest = (interest or "").strip()
    company_description = (company_description or "").strip()

    # If no LLM context, fall back to a minimal message
    if not interest and not company_description:
        return "This company aligns with areas we’re actively tracking in the pipeline."

    name_part = f"Company: {company_name}\n" if company_name else ""
    subcat_part = f"Subcategory: {subcategory}\n" if subcategory else ""

    prompt = (
        "You are a concise VC investment analyst at Montage Ventures. Write a very short blurb (1–2 sentences, max ~45 words) "
        "explaining why you're suggesting a company for the pipeline at Montage. Tie the rationale explicitly to the context you have on the interest from Montage "
        "when relevant. Be specific but brief. Avoid marketing fluff, avoid repeating the company description verbatim, "
        "and do not include headings or bullet points."
    )

    data = (
        f"{name_part}{subcat_part}Interest (from internal discussion):\n{interest}\n\n"
        f"Company Description:\n{company_description}"
    )

    try:
        summary = ask_monty(prompt, data, max_tokens=max_tokens)
        return (summary or "").strip()
    except Exception:
        # Graceful fallback if LLM call fails
        if interest and company_description:
            return (
                "This company fits themes we’re tracking in this space and addresses the problems highlighted in our interest notes."
            )
        if interest:
            return "This company reflects areas we’re actively tracking in this subcategory."
        return "This company aligns with areas we’re actively tracking in the pipeline."


def llm_filter_theme_names(company_description, all_themes, top_k=5):
    theme_names = [theme["thesis_name"] for theme in all_themes]
    theme_list_str = "\n".join(f"- {name}" for name in theme_names)

    prompt = f"""
You are an investment analyst helping match early-stage startups to investment themes.
Given a company description and a list of theme names, choose the {top_k} most relevant themes.
Only return the theme names, one per line, with no explanation.
"""
    data = f"Company Description:\n{company_description}\n\nThemes:\n{theme_list_str}"
    response = ask_monty(prompt, data)
    
    # Clean and extract selected names
    selected = [line.strip("- ").strip() for line in response.splitlines() if line.strip()]
    return set(selected)


def keyword_theme_matching(company_text: str):
    import json
    """
    Enhanced keyword-based matching using DETAILED_THEMES.
    """
    # Load detailed themes
    with open('/Users/matthildur/monty/data/detailed_themes.json', 'r') as f:
        DETAILED_THEMES = json.load(f)

    selected_theme_names = llm_filter_theme_names(company_text, DETAILED_THEMES, top_k=10)

    # Filter full themes to just the selected ones
    filtered_themes = [t for t in DETAILED_THEMES if t["thesis_name"] in selected_theme_names]
    
    matches = []
    company_text_lower = company_text.lower()
    stop_words = {'and', 'the', 'to', 'of', 'in', 'for', 'with', 'on', 'at', 'by', 'from', 'as', 'is', 'are', 'was', 'were'}
    
    for theme in DETAILED_THEMES:
        score = 0
        matched_concepts = []
        context_string = theme.get('thesis_name', '')+ ": "
        
        # Check key concepts
        key_concepts = theme.get('key_concepts', [])
        context_string += f"Key Concepts: {', '.join(key_concepts)}\n\n"
        for concept in key_concepts:
            concept_words = set(concept.lower().split()) - stop_words
            if concept_words and any(word in company_text_lower for word in concept_words):
                score += len(concept_words)
                matched_concepts.append(concept)

        # Check core problems solved
        core_problems_solved = theme.get('core_problems_solved', [])
        context_string += f"Core Problems Solved: {', '.join(core_problems_solved)}\n\n"
        for problem in core_problems_solved:
            problem_words = set(problem.lower().split()) - stop_words
            if problem_words and any(word in company_text_lower for word in problem_words):
                score += len(problem_words)
                matched_concepts.append(problem)

        # Check industry tags
        industry_tags = theme.get('industry_tags', [])
        context_string += f"Industry Tags: {', '.join(industry_tags)}\n\n"
        for tag in industry_tags:
            tag_words = set(tag.lower().split()) - stop_words
            if tag_words and any(word in company_text_lower for word in tag_words):
                score += len(tag_words)
                matched_concepts.append(tag)

        # Check detailed description
        detailed_description = theme.get('detailed_description', '')
        context_string += f"Detailed Description: {detailed_description}\n\n"
        concept_words = set(detailed_description.lower().split()) - stop_words
        if concept_words and any(word in company_text_lower for word in concept_words):
            score += len(concept_words)
            matched_concepts.append(detailed_description)
        
        # Check business model focus (treat as string, not list)
        business_model = theme.get('business_model_focus', '')
        context_string += f"Business Model Focus: {business_model}\n\n"
        if isinstance(business_model, str) and business_model:
            model_words = set(business_model.lower().split()) - stop_words
            if model_words and any(word in company_text_lower for word in model_words):
                score += 3
                matched_concepts.append(f"Business model: {business_model}")
        
        if score >= 5:  # Minimum threshold
            matches.append({
                "theme": theme,  # Store the entire theme object
                "score": score,
                "matched_concepts": matched_concepts,
                "thesis_text": context_string
            })
    
    # Sort by score and return top match
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    if not matches:
        return ()

    return matches


def retrieve_context(query_embedding, query_text, top_n=5):
    from services.database import get_db_connection
    import json
    # Connect to PostgreSQL
    conn = get_db_connection()
    cur = conn.cursor()

    keyword_matches = keyword_theme_matching(query_text)
    keyword_dict = {
        match["thesis_text"]: match["score"]
        for match in keyword_matches
    }

    cur.execute("""
        SELECT thesis_text, embedding <=> %s::vector AS distance
        FROM investment_theses 
        ORDER BY distance
        LIMIT %s
    """, (query_embedding, top_n))
    dense_results = cur.fetchall()

    # Fetch top N relevant Slack context chunks
    cur.execute("""
        SELECT content 
        FROM context_files 
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_embedding, 2*top_n))
    context_results = cur.fetchall()
    cur.close()
    conn.close()

    # Normalize dense scores (smaller = better, so invert)
    dense_texts = [r[0] for r in dense_results]
    dense_distances = [r[1] for r in dense_results]
    max_dist = max(dense_distances) if dense_distances else 1
    dense_scores = {text: 1 - dist / max_dist for text, dist in zip(dense_texts, dense_distances)}

    # Normalize keyword scores
    max_kw = max(keyword_dict.values()) if keyword_dict else 1
    keyword_scores = {text: score / max_kw for text, score in keyword_dict.items()}

    # Combine scores (weighted fusion)
    all_texts = set(dense_scores.keys()).union(keyword_scores.keys())
    combined_scores = {
        text: 0.6 * dense_scores.get(text, 0) + 0.4 * keyword_scores.get(text, 0)
        for text in all_texts
    }

    # Select top-N themes
    top_themes = sorted(combined_scores.items(), key=lambda x: -x[1])[:top_n]

    # Combine context
    relevant_themes = [r[0] for r in top_themes]
    relevant_contexts = [r[0] for r in context_results]
    context_string = (
    "Top Matched Investment Themes:\n\n" +
    "\n\n---\n\n".join(relevant_themes) +  # list of 3 themes
    "\n\nHere is some of the conversation at Montage for more context:\n\n" +
    "\n\n---\n\n".join(relevant_contexts)  # list of context chunks
    )
    return context_string


def ask_monty_with_rag(prompt, row_text, max_tokens=1000, top_n=5):

    # OpenAI for embeddings (no Anthropic equivalent)
    openai_client = OpenAI(api_key=openai_api_key)

    # Embed the query
    combined_string = f"{prompt}\n\n{row_text}"
    embedding_response = openai_client.embeddings.create(model="text-embedding-3-small", input=combined_string)
    query_embedding = embedding_response.data[0].embedding

    context_string = retrieve_context(query_embedding, row_text, top_n)

    # Build the augmented prompt
    augmented_prompt = f"{prompt}\n\nRelevant Context:\n{context_string}\n"

    # Generate Claude response
    claude_client = anthropic.Anthropic(api_key=claude_api_key)
    response = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        system=augmented_prompt,
        messages=[{"role": "user", "content": row_text}],
        max_tokens=max_tokens,
    )

    return response.content[0].text

import asyncio

from agents import Agent, Runner, WebSearchTool, trace
from dotenv import load_dotenv

async def web_search(deal_dict):

    agent = Agent(
        name="Web searcher",
        instructions="You are a helpful agent.",
        tools=[WebSearchTool(user_location={"type": "approximate", "city": "San Francisco"})],
    )

    with trace("Web search example"):
        prompt = f"""
            You must ONLY respond with valid JSON containing exactly one key: "link".

            Task:
            Find the single most relevant public link to an article, blog post, or press release about the **funding announcement** for the company **exactly named** "{deal_dict['Company']}".
            The article must:
            1. Clearly refer to the **same company name** (ignore results for similarly named companies).
            2. Be about the **funding round announcement** (ignore product launches, hiring news, or unrelated topics).
            3. Be **published within the past 7 days** from today.
            4. Match the funding details when possible (Amount: {deal_dict['Amount']}, Round: {deal_dict['Funding Round']}, Investors: {deal_dict['Investors']}).

            Context:
            Company: {deal_dict['Company']}
            Amount: {deal_dict['Amount']}
            Vertical: {deal_dict['Vertical']}
            Funding Round: {deal_dict['Funding Round']}
            Investors: {deal_dict['Investors']}

            Rules:
            - Respond with JSON only, no text, explanation, or commentary.
            - The JSON must be in this exact format:
            {{
            "link": "https://..."
            }}
            - The link must be the direct URL to the correct announcement page.
            - Do not include markdown formatting, code fences, or any other fields.
            - If no matching recent funding announcement is found, return:
            {{
            "link": ""
            }}
            """

        result = await Runner.run(
            agent,
            prompt
        )
        
        # Parse the agent output
        link_json = parse_output(result.final_output)
    
        return link_json
            
def parse_output(output):
    if not output or not isinstance(output, str):
        return {"link": ""}

    cleaned = output.strip()

    # If JSON object exists, try to parse it
    if "{" in cleaned and "}" in cleaned:
        try:
            start_idx = cleaned.find('{')
            end_idx = cleaned.rfind('}') + 1
            json_str = cleaned[start_idx:end_idx]
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # fall through to URL extraction

    # If no valid JSON, try to extract first URL
    url_match = re.search(r'https?://\S+', cleaned)
    if url_match:
        return {"link": url_match.group(0)}

    # Default empty if nothing found
    return {"link": ""}


def fetch_article_content(url, company_name, max_chars=2000):
    """
    Fetch article content from a URL using Parallel API extract.
    
    Args:
        url: URL to extract content from
        company_name: Name of the company (for context)
        max_chars: Maximum characters to return (truncated)
    
    Returns:
        str: Article content excerpt, or empty string if extraction fails
    """
    import os
    
    try:
        api_key = os.getenv("PARALLEL_API_KEY")
        if not api_key:
            return ""
        
        from services.parallel_client import Parallel
        
        parallel_client = Parallel(api_key=api_key)
        
        # Use a highly specific objective to extract only the main article content
        # about the funding announcement, excluding navigation, headers, footers, and UI elements
        objective = f"""Extract ONLY the main article content about {company_name}'s funding announcement. 
        
        INCLUDE:
        - The main article body text about the funding round
        - What the company does (specific product/service description)
        - Funding amount and round type
        - Key investors mentioned
        - Business model details or market context
        - Quotes from founders or investors about the funding
        
        EXCLUDE:
        - Navigation menus, headers, footers
        - Login prompts, sign-up forms
        - Related articles links, sidebar content
        - Social media sharing buttons
        - Website navigation elements
        - Cookie notices, privacy policy links
        - Any UI elements or page structure
        
        Extract only the core article text that describes the funding announcement and the company."""
        
        # Use search queries to further focus on funding-related content
        search_queries = [
            f"{company_name} funding",
            "raised",
            "investors",
            "round"
        ]
        
        extract_result = parallel_client.beta.extract(
            urls=[url],
            objective=objective,
            search_queries=search_queries,
            excerpts=True,
            full_content=False
        )
        
        # Check if extraction was successful
        if extract_result.results and len(extract_result.results) > 0:
            result = extract_result.results[0]
            
            # Extract excerpts from response
            excerpts = []
            if isinstance(result, dict):
                excerpts = result.get('excerpts', [])
            elif hasattr(result, 'excerpts'):
                excerpts = result.excerpts
            
            # Combine excerpts into text
            if excerpts and isinstance(excerpts, list) and len(excerpts) > 0:
                text_content = ' '.join(str(ex) for ex in excerpts if ex)
                
                if text_content and len(text_content.strip()) > 50:
                    # Truncate to max_chars to keep context window manageable
                    return text_content[:max_chars].strip()
        
        return ""
    except Exception as e:
        print(f"  ⚠️  Error extracting article content for {company_name}: {e}")
        return ""


def synthesize_deals(deals_df, max_tokens=1500, fetch_articles=True):
    """
    Synthesize recent deals into 1-3 key insights using OpenAI.
    
    Args:
        deals_df: pandas DataFrame with columns: Company, Amount, Funding Round, Vertical, Category, Link, Investors
        max_tokens: Maximum tokens for the response
        fetch_articles: Whether to fetch article content from links (default: True)
    
    Returns:
        list: List of insight dictionaries, each with:
            - 'pattern': str - The pattern/trend description
            - 'companies': list - List of company names mentioned
            - 'my_take': str - The "My take" analysis
            - 'company_links': dict - Mapping of company names to their article links
    """
    import pandas as pd
    import ast
    import re
    
    if deals_df is None or deals_df.empty:
        return []
    
    # Use all deals (no category filtering)
    filtered_deals = deals_df.copy()
    
    # Format deals data for the prompt
    deals_text = []
    company_link_map = {}
    
    if fetch_articles:
        print(f"  Fetching article content for {len(filtered_deals)} deals...")
    
    for idx, row in filtered_deals.iterrows():
        company = str(row.get("Company", "")).strip()
        amount = str(row.get("Amount", "")).strip() if pd.notna(row.get("Amount")) else "unknown"
        funding_round = str(row.get("Funding Round", "")).strip() if pd.notna(row.get("Funding Round")) else ""
        vertical = str(row.get("Vertical", "")).strip() if pd.notna(row.get("Vertical")) else ""
        category = str(row.get("Category", "")).strip() if pd.notna(row.get("Category")) else ""
        link = str(row.get("Link", "")).strip() if pd.notna(row.get("Link")) and str(row.get("Link")) != "No link found" else ""
        
        # Parse investors
        investors_raw = row.get("Investors", "")
        investors = ""
        if pd.notna(investors_raw) and str(investors_raw).strip() and str(investors_raw).lower() not in ["undisclosed", "unknown", "nan", ""]:
            investors_str = str(investors_raw).strip()
            # Try to parse as Python list
            try:
                if investors_str.startswith('[') and investors_str.endswith(']'):
                    investors_list = ast.literal_eval(investors_str)
                    investors = ', '.join(str(inv) for inv in investors_list)
                else:
                    # Clean up list formatting if present
                    investors = re.sub(r"[\[\]']", "", investors_str)
                    investors = re.sub(r',\s*', ', ', investors)
            except:
                investors = investors_str
        
        if company:
            # Fetch article content if requested and link is available
            article_content = ""
            if fetch_articles and link:
                try:
                    article_content = fetch_article_content(link, company, max_chars=1500)
                    if article_content:
                        print(f"    ✓ Fetched content for {company}")
                    else:
                        print(f"    ⚠️  No content extracted for {company}")
                except Exception as e:
                    print(f"    ⚠️  Error fetching content for {company}: {e}")
                    article_content = ""
            
            # Build deal description
            deal_desc = f"- {company}: {amount} in {funding_round} ({vertical}) - Category: {category}"
            if investors:
                deal_desc += f" - Investors: {investors}"
            if article_content:
                deal_desc += f"\n  Article excerpt: {article_content}"
            
            deals_text.append(deal_desc)
            if link:
                company_link_map[company] = link
    
    if not deals_text:
        return []
    
    deals_list = "\n".join(deals_text)
    
    prompt = """You are a VC investment analyst at Montage Ventures writing a weekly newsletter. 
Analyze ALL the recent deals below and identify 1-3 meaningful themes or patterns that span MULTIPLE deals.

CRITICAL REQUIREMENTS:
- Each insight MUST be based on MULTIPLE deals (at least 2-3 companies). A single deal does not indicate a trend.
- Look for patterns that appear across multiple companies, not isolated cases.
- If you can find relevance to Commerce, Fintech, or Healthcare categories, prioritize those, but insights can span any categories.
- Only create insights if you can identify genuine patterns across multiple deals. It's better to have 1-2 strong insights than 3 weak ones.

AVOID VAGUE LANGUAGE - BE SPECIFIC:
- DO NOT use vague terms like: "AI-driven solutions", "AI technologies", "digital transformation", "innovative platforms", "tech-enabled services"
- DO NOT create insights about broad categories like "the rise of AI" or "automation trends"
- INSTEAD, focus on SPECIFIC use cases, verticals, business models, or market dynamics

GOOD INSIGHT EXAMPLES:
- "Seed rounds for autonomous shopping agents in Commerce (Homie, Claybird) - companies building agents that execute purchases, not just recommend"
- "Healthcare administration tools targeting prior authorization workflows (Taxo, Marit Health) - addressing specific pain points in provider operations"
- "B2B pricing infrastructure for SaaS companies (Dealops, [Company2]) - embedded pricing engines that integrate into sales workflows"

BAD INSIGHT EXAMPLES (too vague):
- "The rise of AI-driven solutions across diverse sectors"
- "Companies leveraging AI technologies"
- "Digital transformation in various industries"

Focus on:
- SPECIFIC verticals or use cases (e.g., "autonomous shopping agents", "prior authorization workflows", "pricing infrastructure")
- SPECIFIC business models or go-to-market approaches
- SPECIFIC market dynamics or pain points being addressed
- Common funding round patterns (e.g., "Seed rounds for X" or "Series Seed for Y")
- Concrete patterns in how companies are solving problems, not abstract technology trends

Use the article excerpts provided to understand what each company actually does. The excerpts contain more detailed information than the brief vertical descriptions - use this to identify specific use cases and business models.

For each insight (1-3 total), provide:
1. The pattern/trend you're observing (must be SPECIFIC and apply to multiple deals - avoid vague generalizations)
2. 2-4 example companies that illustrate this pattern (use exact company names from the list)
3. Your take/analysis on what this means for the market (be concrete and actionable)

Format your response EXACTLY as follows (one insight per block):

INSIGHT 1:
Pattern: [SPECIFIC description of the pattern/trend that spans multiple deals - avoid vague terms]
Companies: [Company1, Company2, Company3]
My take: [concrete analysis of what this means]

[Include INSIGHT 2 and INSIGHT 3 only if you can identify additional SPECIFIC patterns across multiple deals]

Be concise, insightful, specific, and only identify genuine trends that span multiple deals. Avoid vague generalizations."""

    if not deals_text:
        return []
    
    deals_list = "\n".join(deals_text)
    
    data = f"""Recent Deals (with article excerpts where available):
{deals_list}

Identify 1-3 themes/patterns that span MULTIPLE deals and format as specified above. Remember: one deal does not indicate a trend. Use the article excerpts to understand what each company actually does - focus on specific use cases and business models, not vague descriptions."""

    try:
        response = ask_monty(prompt, data, max_tokens=max_tokens, model="claude-sonnet-4-6")

        if not response or not response.strip():
            print("  ⚠️  synthesize_deals: model returned empty response")
            return []

        print(f"  🔍 synthesize_deals raw response (first 500 chars):\n{response[:500]}")
        
        # Parse the response to extract insights
        insights = []
        current_insight = None
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('INSIGHT') or line.startswith('Insight'):
                # Save previous insight if exists
                if current_insight and current_insight.get('pattern'):
                    insights.append(current_insight)
                current_insight = {'pattern': '', 'companies': [], 'my_take': '', 'company_links': {}}
            elif line.startswith('Pattern:') and current_insight:
                current_insight['pattern'] = line.replace('Pattern:', '').strip()
            elif line.startswith('Companies:') and current_insight:
                companies_str = line.replace('Companies:', '').strip()
                # Extract company names (handle brackets, parentheses, and commas)
                companies = [c.strip().strip('[]()') for c in companies_str.split(',') if c.strip()]
                current_insight['companies'] = companies
                # Map companies to their links (fuzzy match for slight variations)
                for company in companies:
                    # Try exact match first
                    if company in company_link_map:
                        current_insight['company_links'][company] = company_link_map[company]
                    else:
                        # Try case-insensitive match
                        for mapped_company, link in company_link_map.items():
                            if company.lower() == mapped_company.lower():
                                current_insight['company_links'][company] = link
                                break
            elif line.startswith('My take:') and current_insight:
                current_insight['my_take'] = line.replace('My take:', '').strip()
            elif current_insight and current_insight.get('my_take') and line and not line.startswith('INSIGHT') and not line.startswith('Insight'):
                # Continue appending to my_take if it spans multiple lines
                current_insight['my_take'] += ' ' + line
        
        # Don't forget the last insight
        if current_insight and current_insight.get('pattern'):
            insights.append(current_insight)

        print(f"  🔍 synthesize_deals: parsed {len(insights)} insights before company-count filter")

        # If parsing failed, try a more flexible approach
        if not insights:
            # Try to extract insights from a more free-form response
            # Split by common separators
            sections = re.split(r'\n\n+|\n---\n|INSIGHT \d+:|Insight \d+:', response)
            for section in sections:
                section = section.strip()
                if not section or len(section) < 50:
                    continue
                
                # Try to extract pattern, companies, and take
                pattern_match = re.search(r'(?:Pattern|Trend|Theme):\s*(.+?)(?:\n|$)', section, re.IGNORECASE)
                companies_match = re.search(r'(?:Companies|Examples):\s*(.+?)(?:\n|My take|$)', section, re.IGNORECASE)
                take_match = re.search(r'My take:\s*(.+?)(?:\n\n|$)', section, re.IGNORECASE | re.DOTALL)
                
                if pattern_match:
                    insight = {
                        'pattern': pattern_match.group(1).strip(),
                        'companies': [],
                        'my_take': take_match.group(1).strip() if take_match else '',
                        'company_links': {}
                    }
                    
                    if companies_match:
                        companies_str = companies_match.group(1).strip()
                        companies = [c.strip().strip('[]()') for c in re.split(r'[,;]', companies_str) if c.strip()]
                        insight['companies'] = companies
                        for company in companies:
                            if company in company_link_map:
                                insight['company_links'][company] = company_link_map[company]
                            else:
                                # Try case-insensitive match
                                for mapped_company, link in company_link_map.items():
                                    if company.lower() == mapped_company.lower():
                                        insight['company_links'][company] = link
                                        break
                    
                    # Only add if pattern exists and has at least 2 companies
                    if insight['pattern'] and len(insight.get('companies', [])) >= 2:
                        insights.append(insight)
        
        # Filter out insights with only one company (not a trend)
        valid_insights = []
        for insight in insights:
            companies = insight.get('companies', [])
            if len(companies) >= 2:
                valid_insights.append(insight)
            else:
                print(f"  🔍 synthesize_deals: dropped insight '{insight.get('pattern','')[:60]}' — only {len(companies)} company/companies")

        print(f"  🔍 synthesize_deals: {len(valid_insights)} valid insights after filter")
        # Limit to 3 insights max
        return valid_insights[:3]
        
    except Exception as e:
        print(f"Error synthesizing deals: {e}")
        return []