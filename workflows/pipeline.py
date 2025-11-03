import json
import pandas as pd
from collections import defaultdict, Counter
from services.notion import import_pipeline
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def load_tree(json_path):
    """Load the investment tree JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)

def format_multi_level_context(children, company_description):
    """Format children and their sub-categories for multi-level context window."""
    lines = []
    for name, child in children.items():
        meta = child.get("meta", {})
        line = f"**{name}**"
        if "description" in meta:
            line += f"\n  Description: {meta['description']}"
        
        # Show sub-categories if they exist
        if "children" in child and child["children"]:
            subcats = list(child["children"].keys())[:5]  # Show first 5 subcategories
            line += f"\n  Sub-categories: {', '.join(subcats)}"
            if len(child["children"]) > 5:
                line += f" (and {len(child['children']) - 5} more)"
        
        lines.append(line)
    return "\n\n".join(lines)

def find_best_node_for_company(node, company_description, path=None):
    """Find the best node for a company using LLM analysis."""
    if path is None:
        path = []
    
    if "children" not in node or not node["children"]:
        return path  # Leaf node, stop here

    multi_level_context = format_multi_level_context(node["children"], company_description)
    
    prompt = f"""
You are an investment analyst categorizing a company into a thematic investment tree.

Company Description:
---
{company_description}
---

Available categories and sub-categories:
---
{multi_level_context}
---

Instructions:
1. Pick the most relevant top-level category (among children), or say "STOP" if you've reached the right level.
2. Choose the category that best matches the company's industry, technology, or business model.

Respond with just the category name, or "STOP" if the current level is appropriate.

Examples:
Fintech
Healthcare  
STOP
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an investment analyst deciding the best category for a company."},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0.2,
            max_tokens=30
        )
        
        choice = response.choices[0].message.content.strip()
        
        if choice == "STOP" or choice not in node["children"]:
            return path
        
        return find_best_node_for_company(node["children"][choice], company_description, path + [choice])
    except Exception as e:
        print(f"  Warning: LLM call failed for company categorization: {e}")
        return path  # Return current path as fallback

def create_company_description(company_row):
    """Create a comprehensive description of the company for LLM analysis."""
    company_name = company_row.get('company_name', 'Unknown Company')
    description = company_row.get('description', '')
    sector = company_row.get('sector', [])
    
    parts = [f"Company: {company_name}"]
    if description and str(description) != 'nan' and description != 'No description':
        parts.append(f"Description: {description}")
    if sector and isinstance(sector, list) and len(sector) > 0:
        parts.append(f"Sectors: {', '.join(sector)}")
    
    return ". ".join(parts)

def analyze_pipeline_companies():
    """Analyze pipeline companies and find their tree paths."""
    # Load the tree
    print("Loading investment tree...")
    tree_json = load_tree("data/taste_tree.json")
    
    # Import pipeline data
    print("Loading pipeline data...")
    pipeline_id = "15e30f29-5556-4fe1-89f6-76d477a79bf8"
    pipeline = import_pipeline(pipeline_id)
    
    print(f"Total companies in pipeline: {len(pipeline)}")
    
    # Filter out companies with priority = "Pass"
    active_pipeline = pipeline[pipeline['priority'] != 'Pass'].copy()
    print(f"Active companies (excluding 'Pass'): {len(active_pipeline)}")
    
    # Analyze each company
    results = []
    print("\nAnalyzing companies...")
    
    for idx, company in active_pipeline.iterrows():
        company_name = company['company_name']
        priority = company['priority']
        
        print(f"  Analyzing: {company_name} (Priority: {priority})")
        
        # Create company description for LLM
        company_description = create_company_description(company)
        
        # Find the best path in the tree
        # Wrap the tree to match expected structure (root with children)
        wrapped_tree = {"children": tree_json}
        best_path = find_best_node_for_company(wrapped_tree, company_description)
        
        results.append({
            'company_name': company_name,
            'priority': priority,
            'founder': company.get('founder', ''),
            'description': company.get('description', ''),
            'sector': company.get('sector', []),
            'tree_path': ' > '.join(best_path) if best_path else 'Root',
            'tree_depth': len(best_path),
            'leaf_category': best_path[-1] if best_path else 'Root'
        })
    
    return pd.DataFrame(results)

def generate_tree_statistics(results_df):
    """Generate statistics about the tree branches in the pipeline."""
    print("\n" + "="*60)
    print("PIPELINE TREE ANALYSIS STATISTICS")
    print("="*60)
    
    # Basic stats
    print(f"\n📊 Basic Statistics:")
    print(f"   Total companies analyzed: {len(results_df)}")
    print(f"   Average tree depth: {results_df['tree_depth'].mean():.1f}")
    print(f"   Max tree depth: {results_df['tree_depth'].max()}")
    print(f"   Companies at root level: {len(results_df[results_df['tree_depth'] == 0])}")
    
    # Priority distribution
    print(f"\n🎯 Priority Distribution:")
    priority_counts = results_df['priority'].value_counts()
    for priority, count in priority_counts.items():
        print(f"   {priority}: {count} companies")
    
    # Tree depth distribution
    print(f"\n🌳 Tree Depth Distribution:")
    depth_counts = results_df['tree_depth'].value_counts().sort_index()
    for depth, count in depth_counts.items():
        level_name = "Root" if depth == 0 else f"Level {depth}"
        print(f"   {level_name}: {count} companies")
    
    # Top tree paths
    print(f"\n🔥 Top Tree Paths:")
    path_counts = results_df['tree_path'].value_counts().head(10)
    for path, count in path_counts.items():
        print(f"   {path}: {count} companies")
    
    # Leaf categories (final categories)
    print(f"\n🍃 Top Leaf Categories:")
    leaf_counts = results_df['leaf_category'].value_counts().head(10)
    for category, count in leaf_counts.items():
        print(f"   {category}: {count} companies")
    
    # Priority vs Tree Path analysis
    print(f"\n🎯 Priority vs Tree Path Analysis:")
    priority_path_analysis = results_df.groupby(['priority', 'tree_path']).size().reset_index(name='count')
    priority_path_analysis = priority_path_analysis.sort_values(['priority', 'count'], ascending=[True, False])
    
    for priority in results_df['priority'].unique():
        priority_data = priority_path_analysis[priority_path_analysis['priority'] == priority]
        print(f"\n   {priority} companies:")
        for _, row in priority_data.head(5).iterrows():
            print(f"     {row['tree_path']}: {row['count']} companies")
    
    return results_df

def save_results(results_df, filename="pipeline_tree_analysis.csv"):
    """Save the results to a CSV file."""
    results_df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to: {filename}")

def main():
    """Main function to run the pipeline analysis."""
    try:
        # Analyze companies
        results_df = analyze_pipeline_companies()
        
        # Generate statistics
        results_df = generate_tree_statistics(results_df)
        
        # Save results
        save_results(results_df)
        
        # Show sample results
        print(f"\n📋 Sample Results:")
        print(results_df[['company_name', 'priority', 'tree_path']].head(10).to_string(index=False))
        
    except Exception as e:
        print(f"Error in pipeline analysis: {e}")
        raise

if __name__ == "__main__":
    main()

