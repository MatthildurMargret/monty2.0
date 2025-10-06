from collections import defaultdict
import ast
import re
from datetime import datetime, timedelta
import pandas as pd

def generate_html(preseed_df, tracking_df, recs, pipeline_dict,
                  greeting_text=None, commerce_df=None,
                  healthcare_df=None, fintech_df=None, test=False):
    """
    Generate the complete HTML newsletter.
    Sections:
      - Greeting
      - Deals (preseeds): old format
      - Tracking: old format
      - Pipeline Insights: stats + founder profiles
    """

    current_date = datetime.now().strftime("%B %d, %Y")

    # ------------------------
    # Deals (original format)
    # ------------------------
    def format_preseeds(df):
        category_display = ["fintech", "commerce", "healthcare", "other"]
        categorized_profiles = defaultdict(list)

        for _, row in df.iterrows():
            company = row["Company"]
            amount = row["Amount"] if pd.notna(row["Amount"]) or str(row["Amount"]) != "nan" else "unknown amount"
            funding_round = row["Funding Round"] if row["Funding Round"] != "Unknown" else ""
            investors = row["Investors"] if (pd.notna(row["Investors"]) and
                                             str(row["Investors"]).lower() not in ["undisclosed", "unknown", "nan"]) else ""
            investors = re.sub(r"^\((.*)\)$", r"\1", investors).strip()
            vertical = row["Vertical"] if row["Vertical"] != "Unknown" else ""
            link = row["Link"] if row["Link"] != "No link found" else ""
            founders = row["Founders"] if pd.notna(row["Founders"]) else ""
            in_newsletter = False

            if founders != "":
                try:
                    founders = ast.literal_eval(founders)
                    for founder in founders:
                        name = founder["Name"]
                        if (
                            commerce_df is not None and name in commerce_df["name"].values
                            or healthcare_df is not None and name in healthcare_df["name"].values
                            or fintech_df is not None and name in fintech_df["name"].values
                        ):
                            in_newsletter = True
                except Exception:
                    pass

            category = row["Category"].lower() if str(row["Category"]).lower() in category_display else "other"
            investors_html = f" (Investors: {investors})" if investors else ""
            founder_html = (
                f" <div><span class='tag-founder'>Founder info and LinkedIn in section below!</span></div>"
                if in_newsletter else ""
            )

            categorized_profiles[category].append(f"""
                <li>
                    <strong><a href="{link}" target="_blank" class="link">{company}</a></strong>  
                    ({vertical}) raised <strong>{amount}</strong> in {funding_round}{investors_html}{founder_html}
                </li>
            """)

        profiles_html = ""
        for category in category_display:
            if categorized_profiles[category]:
                profiles_html += f"<h4>{category}</h4>\n<ul>"
                profiles_html += "\n".join(categorized_profiles[category][:10])  # limit 10 per category
                profiles_html += "</ul>\n"
        return profiles_html

    # ------------------------
    # Tracking (original format)
    # ------------------------
    def format_tracking(df):
        new_updates = df[df['most_recent_update'].notna() & (df['most_recent_update'] != '')]
        last_update = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        if 'last_checked' in new_updates.columns:
            new_updates['last_checked'] = pd.to_datetime(new_updates['last_checked'])
            new_updates = new_updates[new_updates['last_checked'] >= pd.Timestamp(last_update)]
        elif 'most_recent_update_date' in new_updates.columns:
            new_updates['most_recent_update_date'] = pd.to_datetime(new_updates['most_recent_update_date'])
            new_updates = new_updates[new_updates['most_recent_update_date'] >= pd.Timestamp(last_update)]

        profiles_html = ""
        for _, row in new_updates.iterrows():
            company = row["company_name"]
            description = row["most_recent_update"]
            link = row.get("most_recent_update_link", "#")
            if pd.isna(link) or link == '':
                link = "#"

            profiles_html += f"""
                <li>
                    <strong><a href="{link}" target="_blank" class="link">{company}</a></strong>  
                    {description}
                </li>
            """
        return profiles_html

    # ------------------------
    # Pipeline Insights (new unified section)
    # ------------------------
    def format_recs_insights(recs_dict, pipeline_dict):
        def fix_url(url):
            """Ensure URL has proper protocol prefix."""
            if not url or url == '#' or pd.isna(url):
                return '#'
            if not url.startswith('http'):
                return f"https://{url}"
            return url
        
        def format_tags(verticals):
            tags = re.sub(r'\s*\([^)]*\)', '', str(verticals)) if verticals else ""
            tag_list = tags.split(', ') if tags else []
            return ''.join(f"<span class='tag'>{tag}</span>" for tag in tag_list if tag)

        def format_founder_tags(repeat_founder):
            if repeat_founder and str(repeat_founder).lower() in ["yes", "true", "1"]:
                return "<span class='tag-founder'>Repeat founder</span>"
            return ""

        def format_company_tags(companies):
            if isinstance(companies, str) and companies.strip():
                companies = companies.rstrip('.')
                company_list = companies.split(', ')
                return ' '.join(f"<span class='tag tag-company'>{c}</span>" for c in company_list)
            return ""

        def format_thesis_tags(thesis):
            if isinstance(thesis, str) and len(thesis) > 2 and thesis.lower() not in ["negative", "no specific thesis on this space"]:
                return f"<span class='tag-thesis'>{thesis.lower()}</span>"
            return ""

        def clean_location(location):
            if not location or pd.isna(location) or location in ["None", "nan", "Not available"]:
                return ""
            location = re.sub(r'\s+', ' ', str(location).strip())
            if location and not location.startswith("("):
                location = f"({location})"
            return location

        insights_html = "<div class='section'><h3>Pipeline Insights</h3>"
        insights_html += f"<p>Here's a quick overview of the most active areas in the pipeline over the past month. I've found these early-stage companies that operate in similar spaces, check them out! </p>"
        category_order = ["FinTech", "Commerce", "Healthcare", "AI"]
        seen_founders = set()  # Track unique founders across all categories

        for category in category_order:
            if category not in recs_dict or not recs_dict[category]:
                continue

            subcategories = recs_dict[category]
            insights_html += f"<h4>{category}</h4>"

            # Stats overview
            stats = pipeline_dict['subcategories_by_category'].get(category, [])

            # Recommendations per subcategory
            for sub in subcategories:
                founders = sub.get("founders", [])
                if not founders:
                    top_founder = sub.get("top_founder")
                    if top_founder:
                        founder = top_founder

                if not founder:
                    continue

                name = founder.get("name", "")
                company_name = founder.get("company_name", "Unknown")
                
                # Create unique identifier for deduplication
                founder_key = f"{name}|{company_name}"
                
                # Skip if we've already included this founder
                if founder_key in seen_founders:
                    continue
                
                # Only add subcategory header if we have a founder to display
                insights_html += f"<p><strong>{sub['subcategory']}</strong> </p>"
                insights_html += "<ul>"
                
                seen_founders.add(founder_key)
                
                location = clean_location(founder.get("location", ""))
                tags_html = format_tags(founder.get("verticals", ""))
                company_html = format_company_tags(founder.get("company_tags", ""))
                profile_url = fix_url(founder.get("profile_url", "#"))
                short_description = founder.get("product", "")
                thesis_html = format_thesis_tags(founder.get("tree_thesis", ""))
                repeat_tag = format_founder_tags(founder.get("repeat_founder", ""))
                company_website = fix_url(founder.get("company_website", "#"))

                insights_html += f"""
                <li>
                    <div style="display: flex; align-items: center; flex-wrap: wrap;">
                        <strong><a href="{profile_url}" target="_blank" class="link">{name}</a></strong> 
                        <span style="margin-left: 8px;">at <a href="{company_website}" target="_blank" class="link">{company_name}</a> {location}</span>
                    </div>
                    <div style="margin-top: 5px;">
                        {tags_html}{company_html}{repeat_tag}{f'{thesis_html}' if thesis_html else ''}
                    </div>
                    <span class="description">{short_description}</span>
                </li>
                """
                insights_html += "</ul>"

        insights_html += "</div>"
        return insights_html

    # ------------------------
    # Greeting
    # ------------------------
    if not greeting_text:
        greeting_text = "Wishing everyone a happy Friday and a wonderful weekend! – Monty"

    # ------------------------
    # Assemble full HTML
    # ------------------------
    deals_html = format_preseeds(preseed_df)
    tracking_html = format_tracking(tracking_df)
    recs_html = format_recs_insights(recs, pipeline_dict)

    email_body = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Helvetica', 'Arial', sans-serif;
                color: #404040 !important;
                background-color: #ffffff !important;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
            }}
            h2 {{
                background-color: #161F6D;
                color: #ffffff !important;
                font-size: 25px;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 15px;
                text-align: center;
            }}
            .date {{
                font-size: 14px;
                font-weight: bold;
                color: #555 !important;
                text-align: center;
                margin: 10px 0 20px 0;
            }}
            .section {{
                background-color: #ffffff !important;
                padding: 15px;
                margin-bottom: 25px;
                border-radius: 8px;
            }}
            h3 {{
                background: linear-gradient(to right, #F05757, #fde8e8) !important;
                color: #161F6D !important;
                padding: 15px;
                border-radius: 5px;
            }}
            h4 {{
                color: #F05757 !important;
                padding: 5px 0;
                margin-top: 20px;
                margin-bottom: 10px;
            }}
            ul {{
                list-style-type: none;
                padding: 0;
            }}
            li {{
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 1px solid #ddd;
            }}
            li:last-child {{ border-bottom: none; }}
            .link {{
                color: #161F6D !important;
                text-decoration: none;
                font-weight: bold;
                font-size: 14px;
            }}
            .link:hover {{ text-decoration: underline; }}
            .tag, .tag-company, .tag-founder, .tag-thesis {{
                display: inline-block;
                padding: 4px 8px;
                margin: 5px 3px;
                border-radius: 10px;
                font-size: 12px;
                font-weight: bold;
            }}
            .tag {{ background-color: #fde8e8 !important; color: #F05757 !important; }}
            .tag-company {{ background-color: #E3F1FF !important; color: #161F6D !important; }}
            .tag-founder {{ background-color: #f9ffe5 !important; color: #0d3b09 !important; }}
            .tag-thesis {{ background-color: #f9ffe5 !important; color: #4D772F !important; }}
            .description {{
                font-size: 14px;
                color: #555 !important;
                display: block;
                margin-top: 8px;
                line-height: 1.5;
            }}
            .text-box {{
                text-align: center;
                font-size: 14px;
                color: #555 !important;
                background-color: #f9f9f9 !important;
                padding: 10px;
                margin: 10px auto 25px auto;
                width: 60%;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }}
        </style>
    </head>
    <body>
        <h2>Montage Weekly Update</h2>
        <p class="date">{current_date}</p>
        <p class="text-box">{greeting_text}</p>

        <div class="section">
            <h3>Recently Announced Deals</h3>
            {deals_html}
        </div>

        <div class="section">
            <h3>Updates on Tracking</h3>
            <ul>{tracking_html}</ul>
        </div>

        {recs_html}

    </body>
    </html>
    """
    return email_body
