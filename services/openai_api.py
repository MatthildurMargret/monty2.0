from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

def ask_monty(prompt, data, max_tokens=1000):
    client = OpenAI(api_key=openai_api_key)
    # Use the new API to interact with the ChatCompletion model
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Ensure the correct model is specified
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": data}
        ],
        max_tokens=max_tokens,
        temperature=0.5
    )
    return response.choices[0].message.content


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

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Embed the query
    combined_string = f"{prompt}\n\n{row_text}"
    embedding_response = client.embeddings.create(model="text-embedding-3-small", input=combined_string)
    query_embedding = embedding_response.data[0].embedding

    context_string = retrieve_context(query_embedding, row_text, top_n)

    # Build the augmented prompt
    augmented_prompt = f"{prompt}\n\nRelevant Context:\n{context_string}\n"

    # Generate GPT response
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": augmented_prompt},
            {"role": "user", "content": row_text}
        ],
        max_tokens=max_tokens,
        temperature=0.5
    )

    return response.choices[0].message.content

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