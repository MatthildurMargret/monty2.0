from collections import defaultdict
import ast
import re
from datetime import datetime, timedelta
import pandas as pd
from services.openai_api import summarize_company_recommendation

def generate_html(preseed_df, tracking_df, recs, pipeline_dict,
                  greeting_text=None, closing_text=None, commerce_df=None,
                  healthcare_df=None, fintech_df=None, profile_recs=None, test=False):
    """
    Generate the complete HTML newsletter.
    Sections:
      - Greeting
      - Deals (preseeds): old format
      - Tracking: old format
      - Pipeline Insights: stats + founder profiles
      - Closing
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
            
            # Clean up list formatting if present: ['A', 'B'] -> A, B
            if investors:
                # Try to parse as Python list
                try:
                    if investors.startswith('[') and investors.endswith(']'):
                        investors_list = ast.literal_eval(investors)
                        investors = ', '.join(investors_list)
                except:
                    # If parsing fails, use regex to clean up
                    investors = re.sub(r"[\[\]']", "", investors)  # Remove brackets and quotes
                    investors = re.sub(r',\s*', ', ', investors)  # Normalize spacing
            vertical = row["Vertical"] if row["Vertical"] != "Unknown" else ""
            link = row["Link"] if row["Link"] != "No link found" else ""
            founders_raw = row["Founders"] if pd.notna(row["Founders"]) else ""
            
            # Check for founder_linkedin_url column (handle both cases: CSV might have different column names)
            founder_linkedin_url = ""
            if "founder_linkedin_url" in row.index:
                founder_linkedin_url = row["founder_linkedin_url"] if pd.notna(row["founder_linkedin_url"]) else ""
            elif "Founder LinkedIn URL" in row.index:
                founder_linkedin_url = row["Founder LinkedIn URL"] if pd.notna(row["Founder LinkedIn URL"]) else ""
            
            in_newsletter = False
            founders_parsed = None

            if founders_raw != "":
                try:
                    founders_parsed = ast.literal_eval(founders_raw)
                    if isinstance(founders_parsed, list):
                        for founder in founders_parsed:
                            name = founder.get("Name", "") if isinstance(founder, dict) else str(founder)
                            if name and (
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
            
            # Build founder info HTML if we have both founders and LinkedIn URLs
            founder_info_html = ""
            if founders_raw and founder_linkedin_url and str(founder_linkedin_url).strip() and str(founder_linkedin_url).strip().lower() not in ["nan", "none", ""]:
                try:
                    # Parse founders (could be string or list)
                    founder_names = []
                    
                    # Use already parsed founders if available, otherwise parse from raw
                    founders_to_parse = founders_parsed if founders_parsed is not None else founders_raw
                    
                    if isinstance(founders_to_parse, str):
                        # Try to parse as Python list first
                        try:
                            parsed = ast.literal_eval(founders_to_parse)
                            if isinstance(parsed, list):
                                for f in parsed:
                                    if isinstance(f, dict) and "Name" in f:
                                        founder_names.append(f["Name"])
                                    elif isinstance(f, str):
                                        founder_names.append(f)
                            else:
                                # Single string value
                                founder_names = [founders_to_parse]
                        except:
                            # If parsing fails, treat as comma-separated string
                            founder_names = [f.strip() for f in founders_to_parse.split(',') if f.strip()]
                    elif isinstance(founders_to_parse, list):
                        for f in founders_to_parse:
                            if isinstance(f, dict) and "Name" in f:
                                founder_names.append(f["Name"])
                            elif isinstance(f, str):
                                founder_names.append(f)
                    
                    # Only proceed if we have founder names
                    if founder_names:
                        # Parse LinkedIn URLs (comma-separated if multiple)
                        linkedin_urls = [url.strip() for url in str(founder_linkedin_url).split(',') if url.strip() and url.strip().lower() not in ["nan", "none", ""]]
                        
                        # Match founders with LinkedIn URLs (only show if we have both)
                        founder_links = []
                        for i, name in enumerate(founder_names):
                            if i < len(linkedin_urls):
                                url = linkedin_urls[i]
                                # Ensure URL has proper protocol
                                if url and not url.startswith('http'):
                                    url = f"https://{url}"
                                founder_links.append((name, url))
                        
                        # Build HTML for founder info (only if we have at least one name+URL pair)
                        if founder_links:
                            founder_items = []
                            for name, url in founder_links:
                                if url:
                                    founder_items.append(f"<strong>{name}</strong> <a href='{url}' target='_blank' class='link' style='font-size: 12px; color: #161F6D; text-decoration: none;'>LinkedIn</a>")
                            
                            if founder_items:
                                founder_info_html = f"""
                <div style="margin-top: 8px; padding-left: 10px; font-size: 13px; color: #555;">
                    <em>Founder: {', '.join(founder_items)}</em>
                </div>"""
                except Exception:
                    # If parsing fails, silently skip founder info
                    pass

            categorized_profiles[category].append(f"""
                <li>
                    <strong><a href="{link}" target="_blank" class="link">{company}</a></strong>  
                    ({vertical}) raised <strong>{amount}</strong> in {funding_round}{investors_html}{founder_html}{founder_info_html}
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
            """Format company tags, filtering out empty, None, or 'None' values."""
            # Return empty if None, NaN, or empty string
            if not companies or pd.isna(companies):
                return ""
            
            # Convert to string and check for "None" as a string
            companies_str = str(companies).strip()
            if not companies_str or companies_str.lower() in ["none", "nan", ""]:
                return ""
            
            # Clean and split (handle both ", " and "," separators)
            companies_str = companies_str.rstrip('.')
            # Split by comma, then strip each item
            company_list = [item.strip() for item in companies_str.split(',') if item.strip()]
            
            # Filter out empty strings, "None", "nan", etc.
            cleaned_list = []
            for c in company_list:
                c_cleaned = c.strip()
                if c_cleaned and c_cleaned.lower() not in ["none", "nan", ""]:
                    cleaned_list.append(c_cleaned)
            
            # Return formatted tags only if we have valid entries
            if cleaned_list:
                return ' '.join(f"<span class='tag tag-company'>{c}</span>" for c in cleaned_list)
            return ""
        
        def format_school_tags(schools):
            """Format school tags, filtering out empty, None, or 'None' values."""
            # Return empty if None, NaN, or empty string
            if not schools or pd.isna(schools):
                return ""
            
            # Convert to string and check for "None" as a string
            schools_str = str(schools).strip()
            if not schools_str or schools_str.lower() in ["none", "nan", ""]:
                return ""
            
            # Clean and split (handle both ", " and "," separators)
            schools_str = schools_str.rstrip('.')
            # Split by comma, then strip each item
            school_list = [item.strip() for item in schools_str.split(',') if item.strip()]
            
            # Filter out empty strings, "None", "nan", etc.
            cleaned_list = []
            for s in school_list:
                s_cleaned = s.strip()
                if s_cleaned and s_cleaned.lower() not in ["none", "nan", ""]:
                    cleaned_list.append(s_cleaned)
            
            # Return formatted tags only if we have valid entries
            if cleaned_list:
                return ' '.join(f"<span class='tag tag-company'>{s}</span>" for s in cleaned_list)
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

        # Helper: clean and reformat recent deal news text into HTML-safe lines
        def format_recent_news(news_text: str) -> str:
            if not isinstance(news_text, str) or not news_text.strip():
                return ""
            lines = [l.strip() for l in news_text.splitlines()]
            cleaned = []
            seen = set()
            date_line_re = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\]\s*(.+)$")
            for l in lines:
                if not l:
                    continue
                # Skip path/updated header lines like: [Category > Sub] (Updated: 2025-09-10)
                if l.startswith('[') and not date_line_re.match(l):
                    # Likely a path line or other header; drop it
                    continue
                m = date_line_re.match(l)
                if m:
                    date = m.group(1)
                    rest = m.group(2)
                    out = f"{rest} ({date})"
                else:
                    # Fallback: keep as-is if it doesn't match expected formats
                    out = l
                if out not in seen:
                    seen.add(out)
                    cleaned.append(out)
            return '<br>'.join(cleaned)
        
        # Helper: parse and format deals from recent_news for the "Why?" section
        def parse_and_format_deals_for_why_section(news_text: str, months_back: int = 3) -> str:
            """
            Parse deals from recent_news string and format them for the "Why?" section.
            Only includes deals from the past N months.
            
            Args:
                news_text: The recent_news string from tree node
                months_back: Number of months to look back (default: 3)
            
            Returns:
                Formatted string with deals, or empty string if no deals found
            """
            if not isinstance(news_text, str) or not news_text.strip():
                return ""
            
            from datetime import datetime, timedelta
            
            # Calculate cutoff date (3 months ago)
            cutoff_date = datetime.now() - timedelta(days=months_back * 30)
            
            # Parse deal entries - format: [YYYY-MM-DD] Company raised $Amount in Round for Description. Investors: Investors
            # More flexible pattern to handle variations
            deal_pattern = re.compile(
                r'\[(\d{4}-\d{2}-\d{2})\]\s*'  # Date in brackets
                r'([^\[\]]+?)\s+raised\s+'      # Company name (anything before "raised")
                r'([^\s]+(?:\s+[^\s]+)*?)\s+'   # Amount (can have spaces like "$10 M")
                r'in\s+([^f]+?)\s+'              # Round type (everything between "in" and "for")
                r'for\s+([^\.]+?)(?:\.|$)',      # Description (everything after "for" until period or end
                re.IGNORECASE
            )
            
            deals = []
            lines = news_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Skip path/header lines that don't start with date bracket
                if line.startswith('[') and not re.match(r'^\[\d{4}-\d{2}-\d{2}\]', line):
                    continue
                
                # Try to match deal pattern
                match = deal_pattern.search(line)
                if match:
                    date_str = match.group(1)
                    company = match.group(2).strip()
                    amount = match.group(3).strip()
                    # round_type = match.group(4).strip()  # Not used in output but parsed for completeness
                    
                    try:
                        deal_date = datetime.strptime(date_str, '%Y-%m-%d')
                        # Only include deals from past 3 months
                        if deal_date >= cutoff_date:
                            # Calculate relative time
                            days_ago = (datetime.now() - deal_date).days
                            
                            if days_ago < 7:
                                time_str = "last week" if days_ago >= 1 else "this week"
                            elif days_ago < 30:
                                weeks = days_ago // 7
                                time_str = f"{weeks} week{'s' if weeks > 1 else ''} ago"
                            elif days_ago < 90:
                                months = days_ago // 30
                                time_str = f"{months} month{'s' if months > 1 else ''} ago"
                            else:
                                time_str = f"{days_ago // 30} months ago"
                            
                            # Format amount nicely - ensure it has $ if it's a number
                            amount_clean = amount.strip()
                            # Remove extra spaces
                            amount_clean = re.sub(r'\s+', ' ', amount_clean)
                            # Ensure $ prefix if it looks like a number
                            if amount_clean and not amount_clean.startswith('$') and not amount_clean.lower() in ['undisclosed', 'unknown']:
                                # Check if it starts with a number
                                if re.match(r'^\d', amount_clean):
                                    amount_clean = f"${amount_clean}"
                            
                            deals.append({
                                'company': company,
                                'amount': amount_clean,
                                'time': time_str,
                                'date': deal_date
                            })
                    except (ValueError, TypeError) as e:
                        # Skip if date parsing fails
                        continue
            
            # Sort by date (most recent first)
            deals.sort(key=lambda x: x['date'], reverse=True)
            
            # Format deals string
            if deals:
                deal_strings = []
                for deal in deals[:5]:  # Limit to 5 most recent deals
                    deal_strings.append(f"{deal['company']} raised {deal['amount']} {deal['time']}")
                
                return f"Notable investments in this space: {', '.join(deal_strings)}."
            
            return ""

        # Helper: get up to 5 company names from pipeline for a given subcategory
        def get_pipeline_company_names(subcat_name, limit=5):
            try:
                df = pipeline_dict.get('dataframe')
                if df is None or 'subcategory' not in df.columns or 'company_name' not in df.columns:
                    return []
                # Filter by exact subcategory match
                sub_df = df[df['subcategory'] == subcat_name]
                # Prefer most recent if date is present
                if 'date' in sub_df.columns:
                    # Coerce to datetime; errors='coerce' gives NaT for bad values
                    sub_df = sub_df.copy()
                    sub_df['__dt'] = pd.to_datetime(sub_df['date'], errors='coerce')
                    sub_df = sub_df.sort_values(['__dt'], ascending=[False])
                names = sub_df['company_name'].dropna().astype(str).tolist()
                # De-duplicate while preserving order
                seen = set()
                unique_names = []
                for n in names:
                    if n and n not in seen:
                        seen.add(n)
                        unique_names.append(n)
                    if len(unique_names) >= limit:
                        break
                return unique_names
            except Exception:
                return []

        # Compute top 3 subcategories across the pipeline for the summary box, with counts and percentages
        def top_subcategories_stats(pipeline_dict, k=3):
            try:
                all_subs = []
                by_cat = pipeline_dict.get('subcategories_by_category', {}) or {}
                for lst in by_cat.values():
                    if isinstance(lst, list):
                        all_subs.extend(lst)
                # Sort by count desc
                all_subs = sorted(
                    [s for s in all_subs if isinstance(s, dict) and s.get('subcategory')],
                    key=lambda s: s.get('count', 0),
                    reverse=True
                )
                total = int(pipeline_dict.get('total_companies', 0) or 0)
                topk = all_subs[:k]
                # Build entries like: Name (count, X%)
                entries = []
                for s in topk:
                    name = s.get('subcategory', '')
                    count = int(s.get('count', 0) or 0)
                    # If percentage provided use it, otherwise compute
                    pct = s.get('percentage')
                    if pct is None and total > 0:
                        pct = round((count / total) * 100, 1)
                    entries.append({
                        'name': name,
                        'count': count,
                        'percentage': pct
                    })
                return entries, total
            except Exception:
                return [], int(pipeline_dict.get('total_companies', 0) or 0)

        hot_entries, total_companies = top_subcategories_stats(pipeline_dict, 3)
        # Render like: Name (12, 5.3%) • Name (9, 4.1%) • Name (7, 3.2%)
        def render_hot_entry(e):
            name = e.get('name', '')
            count = e.get('count', 0)
            pct = e.get('percentage', None)
            pct_str = f", {pct}%" if pct is not None else ""
            return f"{name} ({count} companies)"
        hot_list = ' • '.join([render_hot_entry(e) for e in hot_entries])
        # Optional small totals line
        filter_date = pipeline_dict.get('filter_date', '')
        totals_line = f"<small>Total companies analyzed since {filter_date}: {total_companies}</small>" if total_companies else ""

        insights_html = "<div class='section'><h3>Sourcing results</h3>"
        #insights_html += f"""
        #<div class="summary-box">
        #<h4>This month’s hottest verticals in the pipeline are:</h4>
        #<p>{hot_list}</p>
        #</div>
        #"""
        insights_html += "<p>Now let's go over some aligned companies I've found over the week. </p>"
        category_order = ["Fintech", "Commerce", "Healthcare", "AI"]
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
                deals = sub.get("deal_activity", [])
                interest = sub.get("interest", "")
                founder = None
                if founders:
                    founder = founders[0]
                else:
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
                # Compose inline list of pipeline company names (top 5)
                inline_names = get_pipeline_company_names(sub['subcategory'])
                inline_str = f" - based on interest in {', '.join(inline_names)}, maybe check out:" if inline_names else ""
                insights_html += f"<p><strong>{sub['subcategory']}</strong>{inline_str} </p>"
                insights_html += "<div class='cards-container'>"
                
                seen_founders.add(founder_key)
                
                # Build location text in the format:  · {location}
                raw_location = founder.get("location", "")
                if raw_location and not pd.isna(raw_location) and str(raw_location) not in ["None", "nan", "Not available"]:
                    location_clean = re.sub(r'\s+', ' ', str(raw_location).strip())
                    location_clean = location_clean.split(",")[0]
                    location = f"Based in {location_clean}"
                else:
                    location = ""
                tags_html = format_tags(founder.get("verticals", ""))
                company_html = format_company_tags(founder.get("company_tags", ""))
                profile_url = fix_url(founder.get("profile_url", "#"))
                short_description = founder.get("product", "")
                thesis_html = format_thesis_tags(founder.get("tree_thesis", ""))
                repeat_tag = format_founder_tags(founder.get("repeat_founder", ""))
                company_website = fix_url(founder.get("company_website", "#"))

                context_blurb = summarize_company_recommendation(
                    interest=interest,
                    company_description=short_description,
                    company_name=company_name,
                    subcategory=sub.get("subcategory", "")
                )

                # Build deal activity box (same style as why-box) if we have recent news (for separate display)
                deal_box_html = ""
                recent_news = ""
                if isinstance(deals, dict):
                    recent_news = deals.get('recent_news', '') or ''
                
                if isinstance(recent_news, str) and recent_news.strip():
                    # Clean, dedupe and reformat dates to end of line
                    deal_content = format_recent_news(recent_news)
                    deal_box_html = f"""
    <div class='why-box'>
        <strong>Recent deal activity:</strong> {deal_content}
    </div>
                    """

                # Build the "Why?" section
                why_section_html = f"<strong>Why {company_name}? </strong> {context_blurb}"

                insights_html += f"""
<div class='company-card'>
    <div class='card-header'>
        <div class='founder-info'>
            <strong><a href="{profile_url}" target="_blank" class="link">{name}</a></strong>
            at <a href="{company_website}" target="_blank" class="link">{company_name}</a>
        </div>
        <div class='description'>
            <span>{location}</span>
        </div>
    </div>
    <div class='card-tags'>
        {tags_html}{company_html}{repeat_tag}{thesis_html}
    </div>
    <p class='description'>{short_description}</p>
    <div class='why-box'>
        {why_section_html}
    </div>
</div>
"""
                insights_html += "</div>"

        insights_html += "</div>"
        return insights_html

    # ------------------------
    # Greeting
    # ------------------------
    if not greeting_text:
        greeting_text = "Wishing everyone a happy Friday and a wonderful weekend! – Monty"
    
    # ------------------------
    # Closing
    # ------------------------
    if not closing_text:
        closing_text = 'Reminder that you can check out all previous recommendations and deals on <a href="https://monty.up.railway.app/" target="_blank" class="link" style="color: #161F6D; text-decoration: underline;">my website</a>.'

    # ------------------------
    # Assemble full HTML
    # ------------------------
    deals_html = format_preseeds(preseed_df)
    tracking_html = format_tracking(tracking_df)
    recs_html = format_recs_insights(recs, pipeline_dict)
    
    # Profile recommendations section
    def format_profile_recs(profiles):
        """Format profile recommendations to match founder card style."""
        if not profiles:
            return ""
        
        def fix_url(url):
            """Ensure URL has proper protocol prefix."""
            if not url or url == '#' or pd.isna(url):
                return '#'
            if not url.startswith('http'):
                return f"https://{url}"
            return url
        
        def format_highlight_tags(highlights):
            """Convert highlights to styled tags."""
            if not highlights:
                return ""
            
            highlight_labels = {
                'potentialToLeave': 'Open to opportunities',
                'priorBackedFounder': 'Prior VC-backed founder',
                'bigTechAlumPublic': 'Big Tech (Public)',
                'bigTechAlumPrivate': 'Big Tech (Private)',
                'employeeDuringIPO': 'IPO Experience',
                'unicornEarlyEngineer': 'Early Unicorn Engineer',
                'topUniversity': 'Top University',
                'employeeDuringMA': 'M&A Experience'
            }
            
            tags_html = ""
            for highlight in highlights:
                if highlight in highlight_labels:
                    tags_html += f"<span class='tag-founder'>{highlight_labels[highlight]}</span>"
            return tags_html
        
        def build_why_explanation(highlights):
            """Build human-readable explanation of why we're recommending."""
            reasons = []
            
            if 'potentialToLeave' in highlights:
                reasons.append("showing signs of being open to new opportunities")
            if 'priorBackedFounder' in highlights:
                reasons.append("previously founded a VC-backed company")
            if 'bigTechAlumPublic' in highlights:
                reasons.append("worked at a major public tech company")
            elif 'bigTechAlumPrivate' in highlights:
                reasons.append("worked at a major private tech company")
            if 'employeeDuringIPO' in highlights:
                reasons.append("experienced an IPO")
            if 'unicornEarlyEngineer' in highlights:
                reasons.append("early engineer at a unicorn")
            if 'topUniversity' in highlights:
                reasons.append("attended a top university")
            
            return ", ".join(reasons) if reasons else "strong background and signals"
        
        profiles_html = "<div class='cards-container'>"
        
        for profile in profiles:
            name = profile.get('fullName', 'Unknown')
            headline = profile.get('headline', 'No headline available')
            location = profile.get('location', 'Location not specified')
            
            # Clean location (just city/state)
            if location and location != 'Location not specified':
                location_parts = location.split(',')
                if len(location_parts) >= 2:
                    location = f"Based in {location_parts[0].strip()}"
                else:
                    location = f"Based in {location}"
            else:
                location = ""
            
            # Get LinkedIn URL
            urls = profile.get('URLs', {})
            linkedin_url = urls.get('linkedin', '')
            profile_url = fix_url(linkedin_url)
            
            # Get highlights
            highlights = profile.get('computed_highlightList', [])
            tags_html = format_highlight_tags(highlights)
            
            # Check if AI provided reasoning
            ai_reason = profile.get('ai_reason', '')
            ai_vertical = profile.get('ai_vertical', '')
            
            if ai_reason:
                # Use AI reasoning - this replaces the computed highlights explanation
                first_name = name.split()[0] if name else 'this person'
                why_text = f"<strong>Why {first_name}?</strong> {ai_reason}"
            else:
                # Fallback to highlight-based explanation (shouldn't happen with AI selection)
                why_text = f"<strong>Why {name.split()[0]}?</strong> {build_why_explanation(highlights).capitalize()}."
            
            profiles_html += f"""
<div class='company-card'>
    <div class='card-header'>
        <div class='founder-info'>
            <strong><a href="{profile_url}" target="_blank" class="link">{name}</a></strong>
        </div>
        <div class='description'>
            <span>{location}</span>
        </div>
    </div>
    <div class='card-tags'>
        {tags_html}
    </div>
    <p class='description'>{headline}</p>
    <div class='why-box'>
        {why_text}
    </div>
</div>
"""
        
        profiles_html += "</div>"
        
        return f"""
        <div class="section">
            <h3>Talent Recommendations</h3>
            <p style="margin-bottom: 15px; color: #555;">Here are some interesting profiles we came across this week:</p>
            {profiles_html}
        </div>
        """
    
    profile_recs_html = format_profile_recs(profile_recs) if profile_recs else ""

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
            .summary-box {{
                background-color: #fff7f7;
                border-left: 4px solid #F05757;
                padding: 10px 15px;
                margin-bottom: 20px;
                border-radius: 5px;
            }}

            .cards-container {{
                display: block;
                margin-top: 10px;
            }}

            .company-card {{
                background: #ffffff;
                border: 1px solid #eee;
                border-radius: 12px;
                padding: 15px;
                margin-bottom: 12px;
            }}

            .card-header {{
                font-size: 14px;
                margin-bottom: 10px;
                display: block;
            }}

            .founder-info {{
                font-weight: bold;
                color: #161F6D !important;
                display: block;
            }}

            .company-info {{
                margin-top: 4px;
                color: #555 !important;
                display: block;
            }}

            .card-tags {{
                margin-bottom: 10px;
                display: block;
            }}

            .why-box {{
                background-color: #f9f9f9;
                border-radius: 6px;
                padding: 8px 10px;
                margin-top: 10px;
                font-size: 13px;
                color: #333 !important;
            }}

            .category-summary {{
                background-color: #fef9e7;
                border-left: 4px solid #ffd54f;
                padding: 10px 15px;
                margin: 25px 0 10px 0;
                border-radius: 5px;
                font-size: 14px;
                color: #333 !important;
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

        {profile_recs_html}

        {recs_html}

        <p class="text-box">{closing_text}</p>

    </body>
    </html>
    """
    return email_body
