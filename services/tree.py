from openai import OpenAI
import os
import json
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

def get_tree_path():
    """Get the correct path to taste_tree.json relative to the project root."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "taste_tree.json")


def load_taste_tree_from_supabase():
    """Load the latest version of taste_tree from Supabase.
    
    Returns:
        dict: The taste tree data (same structure as JSON file)
    
    Raises:
        ImportError: If supabase package is not installed
        ValueError: If required environment variables are not set or no data found
    """
    try:
        from supabase import create_client, Client
    except ImportError:
        raise ImportError(
            "supabase package is required. Install with: pip install supabase"
        )
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url:
        raise ValueError("SUPABASE_URL must be set in your .env file")
    if not supabase_key:
        raise ValueError("SUPABASE_KEY must be set in your .env file")
    
    supabase = create_client(supabase_url, supabase_key)
    
    # Get the latest version of the tree
    response = supabase.table("taste_tree")\
        .select("data")\
        .order("created_at", desc=True)\
        .limit(1)\
        .execute()
    
    if not response.data:
        raise ValueError("No taste_tree data found in Supabase")
    
    return response.data[0]['data']


def load_taste_tree(use_supabase=None):
    """Load taste tree from either Supabase or JSON file.
    
    Args:
        use_supabase: If True, load from Supabase. If False, load from JSON.
                     If None, check TREE_SOURCE env var or default to JSON.
    
    Returns:
        dict: The taste tree data
    """
    # Check environment variable or parameter
    if use_supabase is None:
        tree_source = os.getenv("TREE_SOURCE", "json").lower()
        use_supabase = (tree_source == "supabase")
    
    if use_supabase:
        try:
            return load_taste_tree_from_supabase()
        except Exception as e:
            print(f"⚠️  Error loading from Supabase: {e}")
            print("   Falling back to JSON file...")
            # Fall back to JSON if Supabase fails
            with open(get_tree_path(), 'r') as f:
                return json.load(f)
    else:
        with open(get_tree_path(), 'r') as f:
            return json.load(f)


# Configuration flag to enable/disable tree writes (disabled for Railway deployment)
ENABLE_TREE_WRITES = True

def can_write_tree():
    """Check if tree writes are enabled."""
    return ENABLE_TREE_WRITES

def format_child_context(children):
    """Format metadata summaries for each child category."""
    lines = []
    for name, child in children.items():
        meta = child.get("meta", {})
        line = f"**{name}**"
        if "description" in meta:
            line += f"\n  Vertical Description:\n    {meta['description']}"
        if "notes" in meta:
            line += f"\n  Notes: {meta['notes']}"
        if "recent_news" in meta:
            line += f"\n  Recent News: {meta['recent_news']}"
        lines.append(line)
    return "\n\n".join(lines)

import re

def clean_category_output(choice: str) -> str:
    """
    Normalize and sanitize the model's category output.
    Keeps only a single clean string without labels, punctuation, or filler.
    """
    if not choice:
        return "Unknown"

    # Common prefix removals
    choice = re.sub(r'(?i)(^new\s*category:|^category:|^sub-category:|^suggestion:)', '', choice).strip()

    # If the model outputs a sentence, take just the first phrase (up to ~5 words)
    if len(choice.split()) > 6:
        choice = " ".join(choice.split()[:6])

    # Remove enclosing quotes or punctuation
    choice = choice.strip(" .:-\"'")

    if "None of the" in choice or "stop" in choice.lower():
        return "STOP"

    return choice

def is_invalid_choice(choice: str, children: dict, max_words: int = 6) -> bool:
    """
    Returns True if the LLM output is invalid (too long, multi-line, or contains disallowed phrases).
    """
    normalized = choice.strip().lower()

    # Too long / multi-line
    if "\n" in choice or len(choice.split()) > max_words:
        return True

    # Allowed if it's exactly one of the children
    if normalized in [c.lower() for c in children]:
        return False

    # Otherwise treat as valid candidate new node (let main code handle it)
    return False


def traverse_tree(node, company_description, path=None, trace=None):
    # Initialize mutable defaults to avoid accumulation across calls
    if path is None:
        path = []
    if trace is None:
        trace = []

    # Stop if this node is a leaf
    if "children" not in node or not node["children"]:
        return {
            "path": path,
            "trace": trace,
            "final_status": node.get("meta", {}).get("investment_status", "neutral")
        }

    children = node["children"]
    child_context = format_child_context(children)

    prompt = f"""

Your task is to classify a company by selecting the most appropriate category from a list. 
Each category includes a description to help guide your decision.

Company:
---
{company_description}
---

Sub-categories (with context):
---
{child_context}
---

From an investment perspective, which of these categories is the best fit for this company?

You need to cut through the fluff and think about what the company really does, and from there find the best category for it. 

CRITICAL OUTPUT REQUIREMENT:
Your output MUST BE ONLY ONE STRING containing the chosen category name.
DO NOT write explanations, labels, punctuation, or phrases like 'New category:' or 'None of the categories fit'.
Return ONLY the category name. NOTHING ELSE.
Examples of valid outputs:
- Payments
- AI for Supplier Discovery
- Robotics

Examples of INVALID outputs (DO NOT USE):
- "Category: Payments"
- None of the categories fit
- New suggestion: Robotics
- I think this company belongs in...

IF there is no good match for the company or we are at a sufficiently high level of the tree, return only: STOP

"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an investment analyst at Montage Ventures."},
            {"role": "user", "content": prompt.strip()}
        ],
        temperature=0.2,
        max_tokens=30
    )
    choice = response.choices[0].message.content.strip()
    choice = clean_category_output(choice)

    # Retry if invalid
    retry_count = 0
    while is_invalid_choice(choice, children) and retry_count < 2:
        # Invalid output from model - retrying
        retry_prompt = f"""
    Your last output "{choice}" was invalid.

    Remember: ONLY return a single category name from the list, or STOP.
    Do not add extra words, explanations, or formatting.
    Examples of valid outputs: Payments, Robotics, AI for Supplier Discovery.
    Output must be ≤ 5 words, one line only.
        
    Company:
    ---
    {company_description}
    ---

    Sub-categories:
    ---
    {child_context}
    ---
    """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an investment analyst at Montage Ventures."},
                {"role": "user", "content": retry_prompt.strip()}
            ],
            temperature=0,
            max_tokens=20
        )
        choice = response.choices[0].message.content.strip()
        choice = clean_category_output(choice)
        retry_count += 1

    if choice not in children and not is_invalid_choice(choice, children) and choice != "STOP":

        print("Creating new node at ", path, ": ", choice)
        children[choice] = {
            "meta": {},
            "children": {}
        }
        return {
            "path": path + [choice],
            "trace": trace + [{"category": choice, "status": "neutral", "notes": ""}],
            "final_status": "neutral"
        }
    if choice == "STOP":
        return {
            "path": path,
            "trace": trace,
            "final_status": node.get("meta", {}).get("investment_status", "neutral")
        }

    # Check investment status of chosen node
    chosen_node = children[choice]
    meta = chosen_node.get("meta", {})
    status = meta.get("investment_status", "neutral")
    conviction = meta.get("investment_conviction", "")

    # Append to trace with full metadata
    trace.append({
        "category": choice,
        "status": status,
        "notes": conviction,
        "metadata": meta,
        "depth": len(path) + 1
    })

    # Stop if excluded
    if status == "excluded":
        return {
            "path": path + [choice],
            "trace": trace,
            "final_status": status
        }

    # Recurse if allowed
    return traverse_tree(chosen_node, company_description, path + [choice], trace)


def get_llm_final_recommendation(trace, company_description, founder_info, company_info, investment_theses):
    # Build a rich context showing the decision path
    path_context = []
    path_titles = " > ".join([step['category'] for step in trace if 'category' in step])
    for i, step in enumerate(trace):
        depth_indicator = "  " * i + "→" if i > 0 else ""
        meta = step.get('metadata', {})
        
        # Extract all available metadata
        interest = meta.get('interest', '')
        portfolio = meta.get('portfolio', '')
        investment_status = meta.get('investment_status', '')
        description = meta.get('description', '')
        notes = meta.get('notes', '')
        thesis = meta.get('thesis', '')
        news = meta.get('recent_news', '')
        caution = meta.get('caution', '')
        
        context_parts = []
        if interest: context_parts.append(f"Investment Interest: {interest}")
        if investment_status: context_parts.append(f"Investment Status: {investment_status}")
        if description: context_parts.append(f"Description: {description}")
        if notes: context_parts.append(f"Notes: {notes}")
        if thesis: context_parts.append(f"Thesis: {thesis}")
        if news: context_parts.append(f"Recent News: {news}")
        if caution: context_parts.append(f"Caution: {caution}")
        
        context_str = " | ".join(context_parts) if context_parts else "No additional context"
        
        path_context.append(
            f"{depth_indicator} **{step['category']}** (Status: {step['status']})\n    {context_str}"
        )
    
    # Emphasize the final node
    final_node = trace[-1] if trace else None
    final_emphasis = ""
    if final_node:
        final_meta = final_node.get('metadata', {})
        final_interest = final_meta.get('interest', 'No specific discussion so far on this space')
        final_notes = final_meta.get('notes', 'No specific notes on this space')
        final_thesis = final_meta.get('thesis', 'No specific thesis on this space')
        matching_theme = next(
            (theme for theme in investment_theses if theme.get("thesis_name") == final_thesis),
            None
        )
        if matching_theme:
            description = matching_theme['detailed_description']
            problems = matching_theme['core_problems_solved']
            concepts = matching_theme['key_concepts']
            business_model = matching_theme['business_model_focus']
            theme_details = description + ". Core problems solved: " + ", ".join(problems) + ". Key concepts: " + ", ".join(concepts) + ". Business model focus: " + business_model
        else:
            theme_details = "No specific thesis on this space"
        final_emphasis = f"""
            **FINAL CATEGORY REACHED: {final_node['category']}**
            Investment Thesis for this category: {final_thesis}
            Details on thesis: {theme_details}
            Investment discussion for this category: {final_interest}
            Investment notes for this category: {final_notes}
            Final Status: {final_node['status']}
            """
    else:
        response = "RECOMMENDATION: PASS"
        response += f"\nTHESIS: No specific thesis"
        response += f"\nPATH: No path available"
        return response, trace
    
    path_text = "\n\n".join(path_context)
    
    prompt = f"""
You have evaluated the following company:
---
{company_description}
---

The AI classification system traversed through our investment framework as follows:

**DECISION PATH:**
{path_text}

Note that if the investment discussion contains areas that we are tracking, that means we are interested in this space.

{final_emphasis}

Here is information about the founder:

{founder_info}

And more about the company:

{company_info}

Based on this traversal path, the specific investment context of each category (especially the final category), and the founder information, what is your investment recommendation?
If the founder seems extremely strong, we can be more lenient on the company's criteria. If the company seems highly aligned with Montage's investment thesis, we can be more lenient on the founder's criteria.
Note that 'neutral' investment status along the path is not a bad thing - maybe it's an interesting space that we haven't looked at. If the investment interest is explicitly 'Low', that's a more negative signal.
Important: A past success indication score lower than 7 is not a strong score - a score of 7-9 is the sweet spot. The company tech score gives further indication of how interesting this is from a tech perspective, but it's not the most accurate indicator.
Compare the company description to the context associated with the space and only recommend if the company is aligned with the investment criteria.
Important to remember: 
In all cases, we are only interested in early stage startups with presence in the US, especially in SF Bay Area or California. 
They have to be venture backable - not studios, agencies, consulting firms, services businesses etc. This is important! There has to be a scalable product with a huge market opportunity. 
They should be less than 3 years old - so starting in 2022 or later, and we're not interested if they've already raised over $10M. The earlier the better, and definitely not interested if they raised well over $5M. 

Consider:
1. The investment thesis and interest level, notes at each stage of the path. This is important - if the investment status at each stage is low and then the last node is neutral, it's clearly a space we're not interested in. If there's a lot of "High" along the path but the last node is "Neutral", that still signals strong alignment and maybe it's an interesting new space we haven't seen before.
2. How well the company aligns with the final category's investment criteria. If there is a thesis name corresponding to the final space in the path, it's a strong indicator of alignment.
3. The overall investment status progression through the path. 
4. The founder information: Do they have strong scores, impressive background, any signs that they will lead this company to success? Note that the past success indication score weighs stronger than the industry expertise score, since we are more bullish on founders that are outsiders to the industry and insiders to the problem, meaning that someone with a perfect indsutry experience score might be too established. In fact, an industry score of 10 is not even interesting.

You must make a decision. Choose the most appropriate recommendation based on all available signals, and avoid defaulting to "Neutral" or "Track" unless you have a clear rationale.

Respond in EXACTLY this format:

RECOMMENDATION: [Strong recommend/Recommend/Track/Neutral/Pass]
JUSTIFICATION: [Maximum 4 sentences explaining your reasoning based on the path context. Be specific and include the exact input phrase that justifies the assignment.]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an investment analyst at Montage Ventures."},
            {"role": "user", "content": prompt.strip()}
        ],
        temperature=0.2,
        max_tokens=100
    )
    response = response.choices[0].message.content.strip()
    response += f"\nTHESIS: {final_thesis}"
    response += f"\nPATH: {path_titles}"
    
    return response, trace


def parse_llm_response(response_text):
    """Parse the LLM response to extract recommendation and justification."""
    lines = response_text.strip().split('\n')
    result = {
        'recommendation': 'Unknown', 
        'justification': 'No justification provided',
        'thesis': 'No specific thesis',
        'path': 'No path provided'
    }
    
    for line in lines:
        line = line.strip()
        if line.startswith('RECOMMENDATION:'):
            result['recommendation'] = line.replace('RECOMMENDATION:', '').strip()
        elif line.startswith('JUSTIFICATION:'):
            result['justification'] = line.replace('JUSTIFICATION:', '').strip()
        elif line.startswith('THESIS:'):
            result['thesis'] = line.replace('THESIS:', '').strip()
        elif line.startswith('PATH:'):
            result['path'] = line.replace('PATH:', '').strip()
    
    return result


def add_key_to_all_nodes(node, key):
    """Recursively add 'recent_news' key to all nodes in the tree.
    
    Args:
        node: A node in the tree structure
    """
    # Add recent_news key to current node's metadata if it doesn't exist
    if "meta" in node:
        if key not in node["meta"]:
            node["meta"][key] = ""
    
    # Recursively process children
    if "children" in node and node["children"]:
        for child_name, child_node in node["children"].items():
            add_key_to_all_nodes(child_node, key)


def update_taste_tree_with_key(key):
    """Load taste_tree.json, add recent_news key to all nodes, and save back."""
    import json
    
    # Load the current tree
    with open(get_tree_path(), 'r') as f:
        taste_tree = json.load(f)
    
    print(f"Adding '{key}' key to all nodes...")
    
    # Process each top-level category
    for category_name, category_node in taste_tree.items():
        print(f"Processing category: {category_name}")
        add_key_to_all_nodes(category_node, key)
    
    # Save the updated tree back to file - DISABLED for Railway deployment
    # with open(get_tree_path(), 'w') as f:
    #     json.dump(taste_tree, f, indent=2)
    
    print(f"Would update taste_tree.json with '{key}' keys (write disabled for deployment)")
    print(f"All nodes now have an empty '{key}' field in their metadata.")

def suggest_leaf(trace, company_description):
    path_str = " > ".join([t['category'] for t in trace])
    current_description = trace[-1].get("metadata", {}).get("description", "")

    prompt = f"""

    You have just completed classifying a company using a structured taxonomy. The current final category reached was:

    **{path_str}**

    This category is described as:
    ---
    {current_description}
    ---

    Here is the company description:
    ---
    {company_description}
    ---

    Now, ask yourself: is this category already specific enough to meaningfully group similar companies for investment analysis?
    If YES — that is, if the current category is already the right level of abstraction — respond only with: "none"
    If NO — and you believe a deeper sub-category would be helpful to better characterize this company and other similar companies — suggest a new sub-category name that is:
    - A clear and reusable theme
    - Broad enough to apply to many other companies
    - Not redundant with the existing category
    - Based on the actual function or value proposition (not just company jargon or branding)

    CRITICAL OUTPUT INSTRUCTION:
    Respond only with the sub-category name (e.g., "Bidding Automation", "AI for Supplier Discovery") or "none".
    Return ONLY the bare sub-category name (or bare new sub-category suggestion) with no prefixes, no labels, no quotes, and no punctuation. For example, do NOT write "New sub-category: X" or "Sub-category: X" or "- X" — just write X."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a thematic investment analyst at Montage Ventures."},
            {"role": "user", "content": prompt.strip()}
        ],
        temperature=0.2,
        max_tokens=20
    )
    response = response.choices[0].message.content.strip()
    
    return response

def add_leaf_to_tree(root, path, new_leaf_name, default_meta=None):
    """
    Add a new leaf node to the tree at the end of the given path.

    Parameters:
    - root: the root of the tree
    - path: list of category names representing the path to the parent node
    - new_leaf_name: the name of the leaf to add
    - default_meta: optional metadata to insert into the new leaf
    """
    node = root
    for category in path:
        if "children" not in node or category not in node["children"]:
            raise ValueError(f"Path category '{category}' not found in tree.")
        node = node["children"][category]

    # Ensure "children" dict exists
    if "children" not in node:
        node["children"] = {}

    # Add the new leaf if it doesn't already exist
    if new_leaf_name not in node["children"]:
        node["children"][new_leaf_name] = {
            "meta": default_meta or {},
            "children": {}
        }
        print(f"✅ Added new leaf: {new_leaf_name} under {' > '.join(path)}")
    else:
        print(f"⚠️ Leaf '{new_leaf_name}' already exists under {' > '.join(path)}")

    return node["children"][new_leaf_name]


def analyze_company(company_description, founder_description, company_info, investment_theses):
    with open(get_tree_path(), 'r') as f:
        taste_tree = json.load(f)
        
    new_leafs = []
    # Create a root node structure to match the expected format
    root_node = {
        "meta": {},
        "children": taste_tree
    }

    decision_path = traverse_tree(root_node, company_description)

    if len(decision_path["path"]) < 4 and decision_path['final_status'] != 'excluded':
        #new_leaf = suggest_leaf(decision_path["trace"], company_description)
        new_leaf = "none"
        if new_leaf != "none":
            print("New subcategory suggested: ", new_leaf)
            add_leaf_to_tree(root_node, decision_path["path"], new_leaf)
            decision_path["path"].append(new_leaf)
            decision_path["trace"].append({
                "category": new_leaf,
                "status": "neutral",
                "notes": "",
                "depth": len(decision_path["path"]),
                "path_so_far": decision_path["path"].copy()
            })
            new_leaf_dict = {
                "path": decision_path["path"],
                "leaf": new_leaf
            }
            new_leafs.append(new_leaf_dict)

    final_recommendation_raw, trace = get_llm_final_recommendation(decision_path["trace"], company_description, founder_description, company_info, investment_theses)
    final_recommendation = parse_llm_response(final_recommendation_raw)

    # print("\nFinal recommendation:", final_recommendation['recommendation'])
    # rint("Justification:", final_recommendation['justification'])

    # Tree write disabled for Railway deployment
    # with open(get_tree_path(), 'w') as f:
    #     json.dump(root_node["children"], f, indent=2)
    #print("Tree analysis complete (write to file disabled for deployment)")

    return final_recommendation, trace

def insert_thought_into_tree(thought):
    """Insert a thought/insight into the most relevant node in the taste tree."""
    
    # Load the tree
    with open(get_tree_path(), 'r') as f:
        taste_tree = json.load(f)
    
    # Create a root node structure to match the expected format
    root_node = {
        "meta": {},
        "children": taste_tree
    }
        
    # Find the most relevant node using a modified traversal
    best_node_path, suggested_status = find_best_node_for_thought(root_node, thought)
    print("thought: ", thought)
    print("best_node_path: ", best_node_path)
    print("suggested_status: ", suggested_status)
    
    if not best_node_path and suggested_status:
        print(f"Thought is general — applying investment_status='{suggested_status}' to root.")
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d")
        formatted_thought = f"[{timestamp}] {thought.strip()}"
        # Deduplicate on root meta.interest
        current_notes = root_node['meta'].get('interest', '')
        if _has_thought_already(current_notes, formatted_thought):
            print("  ▷ Skipped root update (duplicate thought)")
        else:
            root_node['meta']['interest'] = (current_notes + ". " + formatted_thought) if current_notes else formatted_thought
        
        #existing_status = root_node['meta'].get('investment_status', '').strip().lower()
        #if not existing_status or existing_status in ['neutral', 'none']:
        #    root_node['meta']['investment_status'] = suggested_status.capitalize()
        #    print(f"Set investment_status to '{suggested_status.capitalize()}' for root node")
        
        # Save updated tree
        if can_write_tree():
            with open(get_tree_path(), 'w') as f:
                json.dump(root_node['children'], f, indent=2)  # Save back the actual tree
            print("Thought insertion complete - tree updated")
        else:
            print("Thought insertion complete (write to file disabled for deployment)")
        return
    elif not best_node_path:
        print("Could not find a suitable node for this thought")
        return

    # Navigate to the target node in the original tree structure
    target_node = taste_tree
    for category in best_node_path:
        if category not in target_node:
            print(f"Error: '{category}' not found in {list(target_node.keys())}")
            return
        target_node = target_node[category]
        
        # If this is not the final node, we need to go into its children
        if category != best_node_path[-1] and 'children' in target_node:
            target_node = target_node['children']
    
    # Get the final node name to find all duplicates
    final_node_name = best_node_path[-1] if best_node_path else None
    
    if not final_node_name:
        print("Could not determine final node name")
        return
    
    # Find all nodes with the same name
    root_node_for_search = {
        "meta": {},
        "children": taste_tree
    }
    duplicate_nodes = find_nodes_by_name(root_node_for_search, final_node_name)
    # Always ensure the exact best_node_path target is included as a fallback
    primary_duplicate = {
        'path': best_node_path,
        'full_path': ' → '.join(best_node_path)
    }
    # Prepend if not already present by path match
    if not any(isinstance(d, dict) and d.get('path') == best_node_path for d in duplicate_nodes):
        duplicate_nodes = [primary_duplicate] + list(duplicate_nodes)
    # Add timestamp for tracking
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d")
    formatted_thought = f"[{timestamp}] {thought}"
    
    nodes_updated = 0
    
    # Insert the thought into all nodes with the same name
    for duplicate in duplicate_nodes:
        # Navigate to each duplicate node
        dup_target_node = taste_tree
        try:
            # Safely obtain path and full_path; compute if missing
            dup_path = None
            dup_full_path = None
            # Interpret different shapes of 'duplicate'
            if isinstance(duplicate, dict):
                dup_path = duplicate.get('path')
                # If path is a string (from find_nodes_by_name), convert to list
                if isinstance(dup_path, str):
                    dup_path = [p.strip() for p in dup_path.split(' > ')]
                dup_full_path = duplicate.get('full_path')
            elif isinstance(duplicate, list):
                # Treat as a path list directly
                dup_path = duplicate
            elif isinstance(duplicate, str):
                # If it's just the name, fall back to the best path
                if duplicate == final_node_name:
                    dup_path = best_node_path
            
            if (not dup_path) and ('node' in duplicate):
                # Attempt to compute path from the node reference
                wrapper_root = {"meta": {}, "children": taste_tree}
                computed = compute_path_to_node(wrapper_root, duplicate['node'])
                if computed:
                    dup_path = computed
                    dup_full_path = dup_full_path or ' → '.join(computed)
            if not dup_path:
                safe_name = duplicate.get('name') if isinstance(duplicate, dict) else final_node_name
                print(f"  ✗ Skipping update for duplicate without path (name={safe_name})")
                continue

            for category in dup_path:
                dup_target_node = dup_target_node[category]
                # If this is not the final node, go into its children
                if category != dup_path[-1] and 'children' in dup_target_node:
                    dup_target_node = dup_target_node['children']
            
            # Insert the thought into the 'notes' field
            current_notes = dup_target_node.get('meta', {}).get('interest', '')
            # Per-node dedup: skip if already present
            if _has_thought_already(current_notes, formatted_thought):
                safe_full_path = dup_full_path or ' → '.join(dup_path)
                print(f"  ▷ Skipped: {safe_full_path} (duplicate thought)")
                continue
            
            updated_notes = (current_notes + ". " + formatted_thought) if current_notes else formatted_thought
            
            # Ensure meta exists
            if 'meta' not in dup_target_node:
                dup_target_node['meta'] = {}
            
            dup_target_node['meta']['interest'] = updated_notes
            nodes_updated += 1
            
            safe_full_path = dup_full_path or ' → '.join(dup_path)
            print(f"  ✓ Updated: {safe_full_path}. {updated_notes}")
            
        except (KeyError, TypeError) as e:
            safe_full_path = duplicate.get('full_path') or (" → ".join(duplicate.get('path', [])) if duplicate.get('path') else final_node_name)
            print(f"  ✗ Failed to update: {safe_full_path} - {e}")
    
        #existing_status = dup_target_node.get('meta', {}).get('investment_status', '').strip().lower()
        #if not existing_status or existing_status in ['neutral', 'none']:
        #    if suggested_status:
        #        dup_target_node['meta']['investment_status'] = suggested_status.capitalize()
        #        print(f"Set investment_status to '{suggested_status.capitalize()}' for {duplicate['full_path']}")
    
    # Save the updated tree back to file
    if can_write_tree():
        with open(get_tree_path(), 'w') as f:
            json.dump(taste_tree, f, indent=2)
        print(f"Thought insertion complete - updated {nodes_updated} nodes")
    else:
        print("Thought insertion complete (write to file disabled for deployment)")
    

def find_best_node_for_thought(node, thought, path=None):
    if path is None:
        path = []
    
    if "children" not in node or not node["children"]:
        return path, None  # Leaf node, stop here

    multi_level_context = format_multi_level_context(node["children"], thought)
    
    prompt = f"""
You are an investment analyst categorizing a new thought into a thematic investment tree.

Thought:
---
{thought}
---

Available categories and sub-categories:
---
{multi_level_context}
---

Instructions:
1. Pick the most relevant top-level category (among children), or say "STOP" if the thought applies broadly.
2. Then assess the investment status this thought implies for that category, using one of:
   - HIGH, MEDIUM, LOW, NEUTRAL, EXCLUDE
Choose the investment status based on the thought's implications for the category. Default to Neutral if it's unclear.

Respond with:
<category name> | <investment_status>

Examples:
Fintech | HIGH  
Healthcare | NEUTRAL  
STOP | LOW
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an investment analyst deciding both the best category and the investment status signal for a new insight."},
            {"role": "user", "content": prompt.strip()}
        ],
        temperature=0.2,
        max_tokens=30
    )
    
    result = response.choices[0].message.content.strip()
    
    if "|" not in result:
        return path, None
    
    # Do NOT uppercase choice — keep as-is to match tree keys
    parts = result.split("|")
    if len(parts) != 2:
        return path, None

    choice = parts[0].strip()  # Keep casing to match tree keys
    status = parts[1].strip().upper()  # Normalize status
    
    if choice == "STOP" or choice not in node["children"]:
        return path, status
    
    return find_best_node_for_thought(node["children"][choice], thought, path + [choice])


def format_multi_level_context(children, thought):
    """Format children and their sub-categories for multi-level context window."""
    
    lines = []
    for name, child in children.items():
        meta = child.get("meta", {})
        line = f"**{name}**"
        
        # Add metadata for the main category
        if meta.get('description', '').strip():
            line += f"\n  Description: {meta['description'][:500]}..."

        if meta.get('interest', '').strip():
            line += f"\n  Recent thoughts: {meta['interest'][:500]}..."
        
        # Add sub-categories preview
        if 'children' in child and child['children']:
            subcategories = list(child['children'].keys())
            if subcategories:
                line += f"\n  Sub-categories: {', '.join(subcategories)}"
                
        else:
            line += "\n  [Leaf Category - No sub-categories]"
        
        lines.append(line)
    
    return "\n\n".join(lines)

def find_duplicate_node_names(tree_root):
    """Find all nodes in the tree that have the same name/title."""
    
    name_to_paths = {}
    
    def collect_all_nodes(node, path=None):
        if path is None:
            path = []
        
        # For each child in this node
        if 'children' in node and node['children']:
            for child_name, child_node in node['children'].items():
                current_path = path + [child_name]
                
                # Add this node name and its path to our collection
                if child_name not in name_to_paths:
                    name_to_paths[child_name] = []
                name_to_paths[child_name].append({
                    'path': current_path.copy(),
                    'full_path': ' → '.join(current_path),
                    'node': child_node
                })
                
                # Recurse into children
                collect_all_nodes(child_node, current_path)
    
    # Start collection from root
    collect_all_nodes(tree_root)
    
    # Find duplicates (names that appear more than once)
    duplicates = {name: paths for name, paths in name_to_paths.items() if len(paths) > 1}
    
    return duplicates, name_to_paths

def find_nodes_by_name(tree_root, target_name):
    """Find all nodes with a specific name."""
    
    _, all_names = find_duplicate_node_names(tree_root)
    return all_names.get(target_name, [])

def compute_path_to_node(tree_root, target_node):
    """Compute the path list to a given target_node by identity.
    Returns a list of category names from root to the node, or None if not found.
    Expects tree_root to be the wrapped root with a 'children' dict.
    """
    def dfs(node, path):
        if 'children' not in node or not node['children']:
            return None
        for name, child in node['children'].items():
            # If identity matches, we found it
            if child is target_node:
                return path + [name]
            # Recurse into children
            found = dfs(child, path + [name])
            if found:
                return found
        return None
    return dfs(tree_root, [])

def _normalize_thought_text(text: str) -> str:
    """Lowercase and strip any leading [YYYY-MM-DD] prefix from a thought string."""
    if not text:
        return ""
    text = text.strip()
    try:
        import re
        text = re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", text).strip()
    except Exception:
        # Fallback: best-effort strip of bracketed date
        if text.startswith("[") and "]" in text:
            text = text.split("]", 1)[1].strip()
    return text.lower()

def _has_thought_already(notes: str, formatted_thought: str) -> bool:
    """Return True if the raw thought already exists in notes, ignoring date prefixes."""
    if not notes:
        return False
    norm_new = _normalize_thought_text(formatted_thought)
    # Split notes on '. ' which is how we join entries
    parts = [p.strip() for p in notes.split('. ') if p.strip()]
    for p in parts:
        if _normalize_thought_text(p) == norm_new:
            return True
    return False

def get_founder_text(row):
    about = row['about'] or "No about information"
    past_success_indication_score = row['past_success_indication_score'] or "No past success indication score"
    startup_experience_score = row['startup_experience_score'] or "No startup experience score"
    repeat_founder = str(row['repeat_founder'])
    technical = str(row['technical'])
    industry_expertise_score = row['industry_expertise_score'] or "Unknown"
    school_tags = row['school_tags'] or "No school tags"
    company_tags = row['company_tags'] or "No company tags"
    company_tech_score = row['company_tech_score'] or "No company tech score"

    repeat_founder = "Yes" in repeat_founder or "true" in repeat_founder
    technical = "Yes" in technical or "true" in technical

    founder_text = f"""About the founder: {about}. 
    Past Success Indication Score (out of 10): {past_success_indication_score}. 
    Startup Experience Score (out of 10): {startup_experience_score}. 
    Industry Expertise Score (out of 10): {industry_expertise_score}. 
    Company Tech Score (out of 10): {company_tech_score}. 
    Background worth noting:
    Schools: {school_tags}. 
    Companies: {company_tags}
    """
    if repeat_founder:
        founder_text += "\nRepeat Founder: Yes"
    if technical:
        founder_text += "\nTechnical: Yes"

    return founder_text

def tree_analysis(profile_dict):
    with open("data/detailed_themes.json", "r") as f:
        investment_theses = json.load(f)
    funding = profile_dict['funding']
    company_name = profile_dict['company_name'] or ''
    description = profile_dict['description_1'] or ''
    product = profile_dict['product'] or ''
    market = profile_dict['market'] or ''
    location = profile_dict['location_1'] or ''
    
    company_text = f"Name: {company_name}" if company_name != '' else ''
    company_text += f"Description: {description}" if description != '' else ''
    company_text += f"Product: {product}" if product != '' else ''
    company_text += f"Market: {market}" if market != '' else ''
    company_text += f"Funding: {funding}" if funding != '' else ''
    founder_text = get_founder_text(profile_dict)

    building_since = profile_dict['building_since'] or ''
    location = profile_dict['location_1'] or ''
    company_info = ""
    if building_since != '':
        company_info += f"Company been around since: {building_since}\n"
    if location != '':
        company_info += f"Location: {location}\n"
    if description != '' and description != 'Not available' and funding_filter(funding):
        try:
            result, trace = analyze_company(company_text, founder_text, company_info, investment_theses)
            rec = result['recommendation']
            justification = result['justification']
            thesis = result['thesis']
            path = result['path']
        except Exception as e:
            print(f"Error analyzing {company_name}: {e}")
    else:
        rec = "skipped"
        justification = ""
        thesis = ""
        path = ""
    
    profile_dict['tree_result'] = rec
    profile_dict['tree_justification'] = justification
    profile_dict['tree_thesis'] = thesis
    profile_dict['tree_path'] = path
    
    return profile_dict

def test_tree():
    import pandas as pd
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from services.database import get_db_connection

    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
    SELECT name, company_name, about, description_1, product, market, location_1, funding, 
    building_since, category, gpt_thesis_check, post_data, ai_reasoning,
    past_success_indication_score, startup_experience_score, company_tech_score, repeat_founder, technical, industry_expertise_score,
    school_tags, company_tags, profile_url, history, location
    FROM founders 
    WHERE founder = true AND (tree_result IS NULL OR tree_result = '')  
    AND product != '' 
    AND history = ''  
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    cursor.close()
    conn.close()
    df = pd.DataFrame(rows, columns=column_names)
    with open("data/detailed_themes.json", "r") as f:
        investment_theses = json.load(f)
    conn = get_db_connection()
    cursor = conn.cursor()
    print("Got ", len(df), " profiles to check")

    for index, row in df.iterrows():
        # Handle None values by converting to empty strings
        funding = row['funding']

        company_name = row['company_name'] or ''
        description = row['description_1'] or ''
        product = row['product'] or ''
        market = row['market'] or ''
        profile_url = row['profile_url'] or ''
        
        company_text = f"Name: {company_name}" if company_name != '' else ''
        company_text += f"Description: {description}" if description != '' else ''
        company_text += f"Product: {product}" if product != '' else ''
        company_text += f"Market: {market}" if market != '' else ''
        company_text += f"Funding: {funding}" if funding != '' else ''
        founder_text = get_founder_text(row)

        building_since = row['building_since'] or ''
        location = row['location'] or ''
        company_info = ""
        if building_since != '':
            company_info += f"Company been around since: {building_since}\n"
        if location != '':
            company_info += f"Location: {location}\n"
        
        if description != '' and description != 'Not available' and funding_filter(funding):
            try:
               result, trace = analyze_company(company_text, founder_text, company_info, investment_theses)
               rec = result['recommendation']
               justification = result['justification']
               thesis = result['thesis']
               path = result['path']
               update_query = """
                UPDATE founders
                SET tree_result = %s,
                    tree_justification = %s,
                    tree_thesis = %s,
                    tree_path = %s
                WHERE profile_url = %s;
            """
               cursor.execute(update_query, (rec, justification, thesis, path, profile_url))
               
               if rec.lower() == 'strong recommend':
                   print("Strong rec: ", profile_url, company_name, justification)
               
            except Exception as e:
                print(f"Error analyzing {company_name}: {e}")
        else:
            print("Skipping", company_name)
            tree_result = 'skipped'
            tree_thesis = ''
            tree_path = ''
            justification = 'skipped'
            update_query = """
                UPDATE founders
                SET tree_result = %s,
                    tree_justification = %s,
                    tree_thesis = %s,
                    tree_path = %s
                WHERE profile_url = %s;
            """
            cursor.execute(update_query, (tree_result, justification, tree_thesis, tree_path, profile_url))

    conn.commit()
    cursor.close()
    conn.close()

def funding_filter(funding):
    if funding is None:
        return True
    if funding == '' or funding == 'NaN' or funding == 'nan':
        return True
    if "Series" in funding and "unknown" not in funding:
        return False
    
    if "seed" in funding.lower():
        return True

    number = funding.split("US$ ")[1]
    number = number.split(", ")[0]
    if "K" in number:
        return True
    if "M" in number:
        if "." in number[:2] or "M" in number[:2]:
            return True
    return False
            

def insert_company_into_tree(company_data):
    """Insert a portfolio company into the most relevant node in the taste tree."""
    
    # Load the taste tree
    with open(get_tree_path(), 'r') as f:
        taste_tree = json.load(f)
    
    # Extract relevant company information
    company_name = company_data.get('Company Name', 'Unknown Company')
    category = company_data.get('Category', '')
    sector = company_data.get('Sector', '')
    brief_description = company_data.get('Brief Description', '')
    status = company_data.get('Status', '')
    
    # Create company description for LLM analysis
    company_description = f"Company: {company_name}. Category: {category}. Sector: {sector}. Description: {brief_description}"
    
    # Create root node structure
    root_node = {
        "meta": {},
        "children": taste_tree
    }
    
    # Find the most relevant node using traversal
    best_node_path = find_best_node_for_company(root_node, company_description)
    
    if not best_node_path:
        print(f"Could not find a suitable node for {company_name}")
        return
    
    # Navigate to the target node
    target_node = taste_tree
    for category_name in best_node_path:
        if category_name not in target_node:
            print(f"Error: '{category_name}' not found in {list(target_node.keys())}")
            return
        target_node = target_node[category_name]
        # If this is not the final node, go into its children
        if category_name != best_node_path[-1] and 'children' in target_node:
            target_node = target_node['children']
    
    # Prepare portfolio entry
    portfolio_entry = {
        "company_name": company_name,
        "brief_description": brief_description,
        "status": status
    }
    
    # Ensure meta exists
    if 'meta' not in target_node:
        target_node['meta'] = {}
    
    # Get existing portfolio or initialize empty list
    current_portfolio = target_node['meta'].get('portfolio', [])
    if isinstance(current_portfolio, str):
        # If portfolio is a string, convert to list
        current_portfolio = [current_portfolio] if current_portfolio.strip() else []
    
    # Get the final node name to find all duplicates
    final_node_name = best_node_path[-1] if best_node_path else None
    
    if not final_node_name:
        print(f"Could not determine final node name for {company_name}")
        return
    
    # Find all nodes with the same name
    root_node_for_search = {
        "meta": {},
        "children": taste_tree
    }
    duplicate_nodes = find_nodes_by_name(root_node_for_search, final_node_name)
    
    nodes_updated = 0
    
    # Insert the company into all nodes with the same name
    for duplicate in duplicate_nodes:
        # Handle the data structure returned by find_nodes_by_name
        if isinstance(duplicate, dict) and 'path' in duplicate:
            # Convert path string to list if needed
            if isinstance(duplicate['path'], str):
                path_list = duplicate['path'].split(' > ')
                full_path_display = duplicate['path']
            else:
                path_list = duplicate['path']
                full_path_display = ' → '.join(path_list)
        else:
            print(f"  ✗ Invalid duplicate node structure: {duplicate}")
            continue
            
        # Navigate to each duplicate node
        dup_target_node = taste_tree
        try:
            for category in path_list:
                if category in dup_target_node:
                    dup_target_node = dup_target_node[category]
                elif 'children' in dup_target_node and category in dup_target_node['children']:
                    dup_target_node = dup_target_node['children'][category]
                else:
                    raise KeyError(f"Category '{category}' not found in tree structure")
            
            # Ensure meta exists
            if 'meta' not in dup_target_node:
                dup_target_node['meta'] = {}
            
            # Get existing portfolio or initialize empty list
            current_portfolio = dup_target_node['meta'].get('portfolio', [])
            if isinstance(current_portfolio, str):
                # If portfolio is a string, convert to list
                current_portfolio = [current_portfolio] if current_portfolio.strip() else []
            
            # Check if company already exists in this node
            company_exists = any(entry.get('company_name') == company_name for entry in current_portfolio if isinstance(entry, dict))
            
            if not company_exists:
                current_portfolio.append(portfolio_entry)
                dup_target_node['meta']['portfolio'] = current_portfolio
                nodes_updated += 1
                
                print(f"  ✓ Added {company_name} to: {full_path_display}")
            else:
                print(f"  ⚠ {company_name} already exists in: {full_path_display}")
                
        except (KeyError, TypeError) as e:
            print(f"  ✗ Failed to add to: {full_path_display} - {e}")
    
    if nodes_updated > 0:
        print(f"✅ Successfully added {company_name} to {nodes_updated} nodes named '{final_node_name}'")
        
        # Save the updated tree
        with open(get_tree_path(), 'w') as f:
            json.dump(taste_tree, f, indent=2)
    else:
        print(f"⚠ {company_name} was not added to any nodes (already exists or errors occurred)")

def find_best_node_for_company(node, company_description, path=None):
    """Find the most relevant node for a portfolio company using multi-level context window."""
    
    if path is None:
        path = []
    
    # Stop if this node is a leaf
    if "children" not in node or not node["children"]:
        return path
    
    # Format multi-level context for LLM
    multi_level_context = format_multi_level_context(node["children"], company_description)
    
    prompt = f"""
Your task is to identify which investment category is most relevant for placing the following portfolio company.

Company Information:
---
{company_description}
---

Available categories with preview of sub-categories:
---
{multi_level_context}
---

Instructions:
1. Consider which category (and its sub-categories) would be the best fit for this portfolio company
2. You can see 2 levels deep - choose the top-level category that seems most relevant
3. If the company fits broadly across multiple subcategories of the current level, respond with "STOP" to place it at the parent level
4. If the company is specific to just ONE subcategory, select that subcategory to go deeper
5. If you want to stop at the current level (not go deeper), respond with "STOP"

Respond with ONLY the category name or "STOP".
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an investment analyst categorizing portfolio companies into the most appropriate investment taxonomy categories."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    choice = response.choices[0].message.content.strip()
    
    # Stop if LLM says to stop or choice is not in children
    if choice == "STOP" or choice not in node["children"]:
        return path    
    # Recurse into the chosen category
    return find_best_node_for_company(node["children"][choice], company_description, path + [choice])

def get_all_nodes():
    import json

    def collect_titles(node, titles):
        for title, content in node.items():
            titles.add(title)
            if 'children' in content:
                    collect_titles(content['children'], titles)

    # Load the JSON file
    with open(get_tree_path(), 'r') as f:
        tree_data = json.load(f)

    # Collect titles
    titles = set()
    collect_titles(tree_data, titles)

    # Print titles
    for title in sorted(titles):
        print(title)

def get_nodes_and_names(use_supabase=None):
    """Get mapping of user names to their assigned category paths.
    
    Args:
        use_supabase: If True, load from Supabase. If False, load from JSON.
                     If None, check TREE_SOURCE env var or default to JSON.
    
    Returns:
        dict: Mapping of user names to lists of category paths, e.g.:
              {"Todd": ["AI > Healthcare > Diagnostics", ...], ...}
    """
    from collections import defaultdict

    def traverse_and_collect(node, path, lead_to_nodes):
        for title, content in node.items():
            new_path = path + [title]
            meta = content.get('meta', {})
            montage_lead = meta.get('montage_lead', '').strip()
            
            if montage_lead:
                leads = [lead.strip() for lead in montage_lead.split(',')]
                for lead in leads:
                    if lead:  # Skip empty strings
                        # Normalize the name to handle case variations
                        normalized_lead = normalize_name(lead)
                        lead_to_nodes[normalized_lead].append(" > ".join(new_path))
            
            if 'children' in content and content['children']:
                traverse_and_collect(content['children'], new_path, lead_to_nodes)

    # Load tree data (from Supabase or JSON)
    tree_data = load_taste_tree(use_supabase=use_supabase)

    # Dictionary to hold the mapping
    lead_to_nodes = defaultdict(list)

    # Traverse the tree
    traverse_and_collect(tree_data, [], lead_to_nodes)

    # Convert defaultdict to regular dict
    lead_to_nodes_dict = dict(lead_to_nodes)
    return lead_to_nodes_dict

def normalize_name(name):
    """Normalize a name to handle case variations and typos.
    
    Maps common variations to canonical names:
    - Daphne, DAphne, daphne -> Daphne
    - Connie, COnnie, connie -> Connie
    - Matthildur, matthildur -> Matthildur
    - etc.
    """
    name = name.strip()
    name_lower = name.lower()
    
    # Map to canonical names
    name_map = {
        'daphne': 'Daphne',
        'connie': 'Connie',
        'matthildur': 'Matthildur',
        'matt': 'Matt',
        'todd': 'Todd',
        'nia': 'Nia'
    }
    
    return name_map.get(name_lower, name)  # Return canonical or original if not found

def find_max_depths(node, path=None, depth=0, results=None):
    if path is None:
        path = []
    if results is None:
        results = {"max_depth": 0, "paths": []}

    # Leaf node
    if "children" not in node or not node["children"]:
        if depth > results["max_depth"]:
            results["max_depth"] = depth
            results["paths"] = [path.copy()]
        elif depth == results["max_depth"]:
            results["paths"].append(path.copy())
        return results

    # Traverse children
    for name, child in node["children"].items():
        find_max_depths(child, path + [name], depth + 1, results)

    return results

def analyze_depth():
    with open(get_tree_path(), 'r') as f:
        taste_tree = json.load(f)

    # Wrap in root node to match expected format
    root_node = {
        "meta": {},
        "children": taste_tree
    }

    # Find and print max depths
    depth_results = find_max_depths(root_node)

    print("🌲 Max depth:", depth_results["max_depth"])
    print("🧭 Deepest paths:")
    for p in depth_results["paths"]:
        print("  > " + " > ".join(p))

def flatten_tree(node, path=None, rows=None, max_depth=5):
    if path is None:
        path = []
    if rows is None:
        rows = []

    meta = node.get("meta", {})
    
    row = {
        f"level_{i+1}": path[i] if i < len(path) else "" for i in range(max_depth)
    }
    row.update({
        "depth": len(path),
        "category": path[-1] if path else "ROOT",
        "description": meta.get("description", ""),
        "notes": meta.get("notes", ""),
        "thesis": meta.get("thesis", ""),
        "interest": meta.get("interest", ""),
        "investment_status": meta.get("investment_status", ""),
        "montage_lead": meta.get("montage_lead", ""),
        "caution": meta.get("caution", ""),
        "news": meta.get("recent_news", ""),
    })
    
    rows.append(row)

    for child_name, child_node in node.get("children", {}).items():
        flatten_tree(child_node, path + [child_name], rows, max_depth=max_depth)

    return rows

def update_tree_from_gsheets():
    import sys
    import os
    from gsheets import read_tree_from_gsheets
    df = read_tree_from_gsheets()
    print(df.columns)

    tree_path = get_tree_path()
    with open(tree_path, "r", encoding="utf-8") as f:
        tree = json.load(f)
    # Wrap raw tree dict so lookups can use a 'children' root as expected
    root = {"children": tree}

    # Normalize and prepare DataFrame
    df = df.fillna("")

    # Build a path list from level_1..level_5 columns (support deeper paths present in sheet)
    level_cols = [c for c in ["level_1", "level_2", "level_3", "level_4", "level_5"] if c in df.columns]

    def row_to_path(row):
        return [str(row[c]).strip() for c in level_cols if str(row[c]).strip()]

    # Map DF columns to tree meta keys (only those present in DF will be written)
    candidate_meta_cols = [
        "investment_status",
        "montage_lead",
    ]
    column_to_meta = {c: c for c in candidate_meta_cols if c in df.columns}

    updates = 0
    missing_paths = 0
    name_only_updates = 0

    for _, row in df.iterrows():
        path_list = row_to_path(row)
        target_nodes = []

        if path_list:
            node = get_node_by_path(root, path_list)
            if node is not None:
                target_nodes = [node]
            else:
                # If path lookup fails but we have a leaf name, update all nodes with that name
                leaf_name = path_list[-1]
                print(f"Missing exact path: {' > '.join(path_list)}")
                target_nodes = find_nodes_by_name(root, leaf_name)
                if target_nodes:
                    name_only_updates += 1
                else:
                    missing_paths += 1
        else:
            missing_paths += 1

        for node in target_nodes:
            node.setdefault("meta", {})
            wrote_any = False
            for df_col, meta_key in column_to_meta.items():
                value = row[df_col]
                if isinstance(value, str):
                    value = value.strip()
                    try:
                        prev = node["meta"][meta_key]
                    except KeyError:
                        prev = ""
                if value != "":
                    if prev != value:
                        node["meta"][meta_key] = value
                        print(f"Updated {meta_key} for {row['category']}: {value}")
                        wrote_any = True
                else:
                    if prev != "":
                        # node["meta"][meta_key] = ""
                        print(f"Cleared {meta_key} for {row['category']}: {prev}")
                        # wrote_any = True
            if wrote_any:
                updates += 1

    if updates > 0:
        # Tree write disabled for Railway deployment
        # with open(tree_path, "w", encoding="utf-8") as f:
        #     json.dump(tree, f, ensure_ascii=False, indent=2)
        print(f"Would save {updates} updates to tree (write disabled for deployment)")
        print("Found some updates")
    print(f"Updated {updates} nodes. Missing path matches: {missing_paths}. Name-only updates applied for {name_only_updates} rows.")

    
def get_node_by_path(root_node, path_list):
    node = root_node
    for name in path_list:
        if not name:
            continue
        children = node.get("children", {})
        # match by exact key first
        if name in children:
            node = children[name]
            continue
        # fallback: case/whitespace-insensitive match
        normalized = name.strip().lower()
        key_map = {k.strip().lower(): k for k in children.keys()}
        if normalized in key_map:
            node = children[key_map[normalized]]
        else:
            return None
    return node    

def find_nodes_by_name(root_node, name):
    target = name.strip().lower()
    matches = []
    def _walk(node, path=[]):
        # node is a dict with possible children
        for child_name, child in node.get("children", {}).items():
            current_path = path + [child_name]
            if child_name.strip().lower() == target:
                matches.append({"node": child, "path": " > ".join(current_path)})
            _walk(child, current_path)
    _walk(root_node)
    return matches

def find_similar_nodes(root_node, name):
    target = name.strip().lower()
    matches = []
    def _walk(node, path=[]):
        # node is a dict with possible children
        for child_name, child in node.get("children", {}).items():
            current_path = path + [child_name]
            if target in child_name.strip().lower() or child_name.strip().lower() in target:
                matches.append({"node": child, "path": " > ".join(current_path)})
            _walk(child, current_path)
    _walk(root_node)
    return matches


# Uncomment the line below to add 'recent_news' key to all nodes in taste_tree.json
#update_taste_tree_with_key("last_updated")

# Example usage:
# description = "Deploying satellites to space"
# analyze_company(description)

#update_tree_from_gsheets()

# Test the analysis on data from database
# test_tree()
# get_all_nodes()
# get_nodes_and_names()
#with open(get_tree_path()) as f:
#    tree = json.load(f)
#root = {"meta": {}, "children": tree}
#rows = flatten_tree(root)
#df = pd.DataFrame(rows)
#df.to_csv("tree_flat_export.csv", index=False)

#node_name = 'Robotics'
#matches = find_similar_nodes(root, node_name)
#print(f"Found {len(matches)} instances of '{node_name}':")
#for match in matches:
#    print(f"Path: {match['path']}")
#    print("---")


#thoughts_file = '/Users/matthildur/monty/data/slack_context/misc.txt'
#i = 0
#with open(thoughts_file, 'r') as f:
#    thoughts = f.readlines()
#    # Only process the last 20 thoughts
#    last_20_thoughts = thoughts[-30:] if len(thoughts) > 30 else thoughts
#    print(last_20_thoughts)
#    for thought in last_20_thoughts:
#        insert_thought_into_tree(thought)


# Insert portfolio companies into the tree
#portfolio_data = pd.read_csv('data/portfolio.csv')
#for index, row in portfolio_data.iterrows():
#    insert_company_into_tree(row)

# investment_status = ["high", "neutral", "cautious", "deprioritized", "excluded"]




        