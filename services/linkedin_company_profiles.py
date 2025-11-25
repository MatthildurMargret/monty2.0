from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import re
import time


def extract_overview(soup):
    """Extract the company overview section."""
    overview_tag = soup.find('p', class_='break-words white-space-pre-wrap t-black--light text-body-medium')
    if overview_tag:
        return f"Overview: {overview_tag.get_text(strip=True)}\n\n"
    return ""


def extract_company_title(soup):
    overview_tag = soup.find('p', class_='org-top-card-summary__tagline')
    if overview_tag:
        return f" {overview_tag.get_text(strip=True)}"
    return ""


def extract_funding_details(soup):
    """Extracts last funding round details from the page and returns a dictionary."""
    funding_data = {
        "amount_raised": None,  # Funding amount
        "round_details": None  # Round type, date, and investors
    }

    # Locate the funding section
    funding_section = soup.find('div', class_='org-funding__card-spacing')
    if not funding_section:
        return funding_data  # Return empty if no funding info is found

    # Extract Last Round Type & Date
    round_info = funding_section.find('span', class_='org-funding__funding-type')
    round_date = round_info.find_next_sibling(string=True) if round_info else None
    round_text = f"{round_info.get_text(strip=True)} {round_date.strip()}" if round_info and round_date else None

    # Extract Funding Amount
    amount_section = funding_section.find('span', class_='t-24 t-light t-black--light')
    amount_raised = amount_section.get_text(strip=True) if amount_section else None

    # Extract Investors
    investors = []
    investors_section = funding_section.find('ul', class_='list-style-none')
    if investors_section:
        investor_names = investors_section.find_all('span', class_='v-align-middle t-14 t-black--light t-normal pl2')
        investors = [investor.get_text(strip=True) for investor in investor_names]

        # Check for additional investors (+X others)
        additional_investors = investors_section.find('div', class_='org-funding__other-investors-icon')
        if additional_investors:
            investors.append(additional_investors.get_text(strip=True))

    # Populate the dictionary
    funding_data["amount_raised"] = amount_raised
    funding_data["round_details"] = {
        "round": round_text,
        "investors": investors
    }

    return funding_data


def extract_details(soup):
    """Extract key-value details (e.g., website, industry, etc.)."""
    details = soup.find_all('dl')
    company_info = ""
    website = ""  # Initialize website variable
    
    for detail in details:
        terms = detail.find_all('dt')
        definitions = detail.find_all('dd')

        for term, definition in zip(terms, definitions):
            key = term.get_text(strip=True).lower()
            value = definition.get_text(strip=True)

            if 'website' in key:
                link = definition.find('a')
                if link and link.has_attr('href'):
                    website = link.get('href')
                else:
                    website = value
                company_info += f"Website: {website}\n "
            elif 'industry' in key:
                company_info += f"Industry: {value}\n"
            elif 'company size' in key:
                company_info += f"Company Size: {value}\n "
            elif 'headquarters' in key:
                company_info += f"Headquarters: {value}\n "

    return company_info, website


def extract_locations(soup):
    """Extract all company location details."""
    location_cards = soup.find_all('div', class_='org-location-card')
    if not location_cards:
        return ""

    locations_info = "Locations: "
    for card in location_cards:
        name = card.find('h4', class_='t-14 t-black--light t-bold mr1 mt1')
        address = card.find('p', class_='t-14 t-black--light t-normal break-words')
        if name and address:
            locations_info += f"  - {name.get_text(strip=True)}, {address.get_text(strip=True)}; "
    return locations_info


def get_detailed_company_description(experience_data, company_name, company_description, context=None):

    better_description = ""
    funding_data = {
        "amount_raised": None,  # Funding amount
        "round_details": None  # Round type, date, and investors
    }
    funding = ""
    website = ""  # Initialize website variable
    if "stealth" not in company_name.lower():
        try:
            better_description, funding_data, website = scrape_extra_company_info_playwright(experience_data, context)  # Fixed function call
        except Exception as e:
            print(f"Error scraping company info: {e}")
        if better_description != "":
            if company_description != "Not available":
                company_description = company_description + ". " + better_description
            else:
                company_description = better_description

        if funding_data["amount_raised"]:
            investors = ", ".join(funding_data["round_details"]['investors']) if funding_data["round_details"]['investors'] else ""
            funding = (
                f"{funding_data['amount_raised']}, {funding_data['round_details']['round']}, {investors}"
            )
    else:
        return company_description, "", ""  # Return empty website as well

    return company_description, funding, website


def scrape_extra_company_info_playwright(experience_data, context=None):
    company_link = experience_data[0]['link']
    if context:
        page = context.new_page()
        page.goto(company_link, timeout=60000)
        company_html = page.content()
        page.close()
    else:
        company_html = ""
    soup = BeautifulSoup(company_html, 'html.parser')
    details, website = extract_details(soup)

    # Gather all company information
    company_info = ""
    company_info += extract_overview(soup)
    company_info += extract_company_title(soup)
    company_info += details
    company_info += extract_locations(soup)

    funding_info = extract_funding_details(soup)

    # If website wasn't found in extract_details, try alternative methods
    if not website:
        # Try to find website in contact info section
        contact_info = soup.find('section', {'class': 'artdeco-card p5 mb4'})
        if contact_info:
            website_link = contact_info.find('a', href=True)
            if website_link and 'http' in website_link.get('href'):
                website = website_link.get('href')
        
        # Try to find website in the page content
        if not website:
            website_links = soup.find_all('a', href=True)
            for link in website_links:
                href = link.get('href')
                if href and 'http' in href and 'linkedin.com' not in href:
                    website = href
                    break

    # Format the final string
    return " ".join(line.strip() for line in company_info.splitlines()), funding_info, website

def parse_website(page):
    # Extract the page content
    html_content = page.content()
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract title and meta description
    title = soup.title.string.strip() if soup.title else ""
    if title:
        all_extracted_info.append(f"Website Title: {title}")
    
    meta_description = soup.find('meta', attrs={'name': 'description'})
    if not meta_description:
        meta_description = soup.find('meta', attrs={'property': 'og:description'})
    
    if meta_description and meta_description.get('content'):
        all_extracted_info.append(f"Meta Description: {meta_description['content'].strip()}")
    
    # Extract main content text
    main_content = extract_main_content(soup)
    if main_content:
        all_extracted_info.append(f"Main Content: {main_content[:500]}...")
    
    # Extract key features
    features = extract_features(soup)
    if features:
        all_extracted_info.append(f"Features: {', '.join(features)}")
    
    # Extract target audience
    target_audience = extract_target_audience(soup, main_content)
    if target_audience:
        all_extracted_info.append(f"Target Audience: {target_audience}")
    
    # Collect all navigation links for analysis
    nav_links = collect_navigation_links(soup, website)
    if nav_links:
        all_extracted_info.append(f"Navigation Links: {', '.join(nav_links.keys())}")
    
    # Look for "About" page
    about_link = find_link_by_keywords(nav_links, ['about', 'about us', 'our company', 'who we are', 'mission'])
    if about_link:
        try:
            # Add a small delay to avoid being detected as a bot
            time.sleep(1)
            
            # Navigate to the About page
            page.goto(about_link, wait_until="domcontentloaded")
            page.wait_for_load_state("networkidle", timeout=10000)
            
            # Extract content from About page
            about_html = page.content()
            about_soup = BeautifulSoup(about_html, 'html.parser')
            about_content = extract_main_content(about_soup)
            if about_content:
                all_extracted_info.append(f"About Page Content: {about_content[:500]}...")
        except Exception as e:
            print(f"Error accessing About page: {e}")
    
    # Look for "Products" or "Solutions" page
    product_link = find_link_by_keywords(nav_links, ['product', 'products', 'solution', 'solutions', 'services', 'platform', 'features'])
    if product_link:
        try:
            # Add a small delay to avoid being detected as a bot
            time.sleep(1)
            
            # Navigate to the Products page
            page.goto(product_link, wait_until="domcontentloaded")
            page.wait_for_load_state("networkidle", timeout=10000)
            
            # Extract content from Products page
            product_html = page.content()
            product_soup = BeautifulSoup(product_html, 'html.parser')
            product_content = extract_product_info(product_soup)
            if product_content:
                all_extracted_info.append(f"Product Page Content: {product_content[:500]}...")
                
            # Try to find product categories or specific products on this page
            product_links = extract_product_links(product_soup, product_link)
            if product_links:
                all_extracted_info.append(f"Product Categories: {', '.join(product_links.keys())}")
        except Exception as e:
            print(f"Error accessing Products page: {e}")
    
    # Look for product information on the main page
    product_info = extract_product_info(soup)
    if product_info:
        all_extracted_info.append(f"Product Info from Main Page: {product_info[:500]}...")
    
    # Check for "Pricing" page
    pricing_link = find_link_by_keywords(nav_links, ['pricing', 'plans', 'packages', 'subscription'])
    if pricing_link:
        try:
            time.sleep(1)
            page.goto(pricing_link, wait_until="domcontentloaded")
            page.wait_for_load_state("networkidle", timeout=10000)
            
            pricing_html = page.content()
            pricing_soup = BeautifulSoup(pricing_html, 'html.parser')
            pricing_info = extract_pricing_info(pricing_soup)
            if pricing_info:
                pricing_text = []
                for tier in pricing_info:
                    if 'name' in tier and 'price' in tier:
                        pricing_text.append(f"{tier['name']}: {tier['price']}")
                    elif 'info' in tier:
                        pricing_text.append(tier['info'])
                
                if pricing_text:
                    all_extracted_info.append(f"Pricing Information: {', '.join(pricing_text)}")
        except Exception as e:
            print(f"Error accessing Pricing page: {e}")
    
    return all_extracted_info, page


def analyze_company_website(website, company_description="", page=None):
    """
    Analyze a company website to extract information about their products, services, and mission.
    
    Args:
        website (str): URL of the company website
        company_description (str, optional): LinkedIn company description to enhance the analysis
        
    Returns:
        str: Extracted information about the company
    """
    if not website:
        return ""
    
    # Ensure the website URL is properly formatted
    if not website.startswith('http'):
        website = 'https://' + website
    
    all_extracted_info = []
    all_extracted_info.append(f"Website URL: {website}")
    
    # Add LinkedIn description if provided
    if company_description:
        all_extracted_info.append(f"LinkedIn Description: {company_description}")
    
    if page:
        all_info, page = parse_website(page)
        all_extracted_info.extend(all_info)
    else:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
                )
                page = context.new_page()
                
                # Set a reasonable timeout
                page.set_default_timeout(15000)
                
                try:
                    all_info = parse_website(page)
                    all_extracted_info.extend(all_info)
                except Exception as e:
                    print(f"Error navigating to website: {e}")
                
                # Close browser
                browser.close()
        
        except Exception as e:
            print(f"Error analyzing website: {e}")
            return ""
    
    # Join all extracted information into a single string
    return "\n\n".join(all_extracted_info)


def collect_navigation_links(soup, base_url):
    """Collect all navigation links from the website."""
    nav_links = {}
    
    # Look for navigation elements
    nav_elements = soup.select('nav, .nav, .navigation, .menu, header, .header')
    
    for nav in nav_elements:
        links = nav.find_all('a')
        for link in links:
            text = link.get_text(strip=True)
            href = link.get('href')
            
            if text and href:
                # Skip very short text or non-descriptive links
                if len(text) < 3 or text.lower() in ['home', 'logo']:
                    continue
                
                # Construct full URL if relative
                if not href.startswith('http'):
                    if href.startswith('/'):
                        href = base_url.rstrip('/') + href
                    else:
                        href = base_url.rstrip('/') + '/' + href
                
                nav_links[text.lower()] = href
    
    # Also check for footer links which often contain important pages
    footer_elements = soup.select('footer, .footer')
    for footer in footer_elements:
        links = footer.find_all('a')
        for link in links:
            text = link.get_text(strip=True)
            href = link.get('href')
            
            if text and href and len(text) > 2:
                # Construct full URL if relative
                if not href.startswith('http'):
                    if href.startswith('/'):
                        href = base_url.rstrip('/') + href
                    else:
                        href = base_url.rstrip('/') + '/' + href
                
                nav_links[text.lower()] = href
    
    return nav_links


def find_link_by_keywords(nav_links, keywords):
    """Find a link in the navigation that matches any of the keywords."""
    for text, href in nav_links.items():
        for keyword in keywords:
            if keyword.lower() in text.lower():
                return href
    return None


def extract_product_links(soup, base_url):
    """Extract links to specific products or product categories."""
    product_links = {}
    
    # Look for product cards or product listings
    product_cards = soup.select('.product-card, .card, .product-item, .product-box, .feature-box')
    for card in product_cards:
        link = card.find('a')
        if link:
            name = link.get_text(strip=True)
            href = link.get('href')
            
            if name and href:
                # Construct full URL if relative
                if not href.startswith('http'):
                    if href.startswith('/'):
                        href = base_url.rstrip('/') + href
                    else:
                        href = base_url.rstrip('/') + '/' + href
                
                product_links[name] = href
    
    # If no product cards found, look for lists or sections that might contain products
    if not product_links:
        # Look for headings that might indicate product sections
        product_headings = soup.find_all(['h2', 'h3'], string=lambda s: s and any(keyword in s.lower() for keyword in ['product', 'solution', 'service', 'platform', 'software', 'app', 'application']))
        
        for heading in product_headings:
            # Find the closest container that might contain product links
            container = heading.find_next(['div', 'section', 'ul'])
            if container:
                links = container.find_all('a')
                for link in links:
                    name = link.get_text(strip=True)
                    href = link.get('href')
                    
                    if name and href and len(name) > 3:
                        # Construct full URL if relative
                        if not href.startswith('http'):
                            if href.startswith('/'):
                                href = base_url.rstrip('/') + href
                            else:
                                href = base_url.rstrip('/') + '/' + href
                        
                        product_links[name] = href
    
    return product_links


def extract_pricing_info(soup):
    """Extract pricing information from a pricing page."""
    pricing_info = []
    
    # Look for pricing tables
    pricing_tables = soup.select('.pricing-table, .pricing, .plans, #plans, .packages, #packages')
    if pricing_tables:
        for table in pricing_tables:
            # Look for pricing tiers
            tiers = table.select('.pricing-tier, .plan, .package, .card, .col')
            for tier in tiers:
                tier_info = {}
                
                # Try to find the tier name
                tier_name = tier.find(['h2', 'h3', 'h4', '.tier-name', '.plan-name'])
                if tier_name:
                    tier_info['name'] = tier_name.get_text(strip=True)
                
                # Try to find the price
                price = tier.select_one('.price, .cost, .amount')
                if price:
                    tier_info['price'] = price.get_text(strip=True)
                
                # If we have at least a name or price, add it
                if tier_info:
                    pricing_info.append(tier_info)
    
    # If no structured pricing found, try to extract from text
    if not pricing_info:
        pricing_patterns = [
            r'(\$\d+(?:\.\d+)?(?:/[a-zA-Z]+)?)',  # $X or $X.XX or $X/month
            r'((?:free|basic|premium|enterprise|starter|professional|business|team)(?:\s+plan)?(?:\s*:\s*\$\d+(?:\.\d+)?(?:/[a-zA-Z]+)?)?)',  # Plan names with optional price
        ]
        
        text = soup.get_text()
        for pattern in pricing_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                pricing_info.append({'info': match.strip()})
    
    return pricing_info


def format_company_info(company_info):
    """Format the extracted company information into a readable summary."""
    summary = []
    
    # Add title
    if company_info.get('title'):
        summary.append(f"Website Title: {company_info['title']}")
    
    # Add meta description
    if company_info.get('meta_description'):
        summary.append(f"Description: {company_info['meta_description']}")
    
    # Add LinkedIn description if available
    if company_info.get('linkedin_description'):
        summary.append(f"LinkedIn Description: {company_info['linkedin_description']}")
    
    # Extract key information from content
    content_to_analyze = ""
    if company_info.get('linkedin_description'):
        content_to_analyze += company_info['linkedin_description'] + " "
    if company_info.get('about_content'):
        content_to_analyze += company_info['about_content'] + " "
    if company_info.get('main_content'):
        content_to_analyze += company_info['main_content'] + " "
    
    # Add product categories if found
    if company_info.get('product_categories'):
        categories = company_info['product_categories']
        summary.append(f"Product Categories: {', '.join(categories)}")
    
    # Add detailed product information
    if company_info.get('product_details') and isinstance(company_info['product_details'], dict):
        product_details = company_info['product_details']
        summary.append("Products/Services:")
        for name, desc in product_details.items():
            summary.append(f"• {name}: {desc}")
    else:
        # Add product information from other sources
        product_info = ""
        if company_info.get('product_details') and isinstance(company_info['product_details'], str):
            product_info += company_info['product_details'] + " "
        if company_info.get('product_info'):
            product_info += company_info['product_info'] + " "
        
        if product_info:
            # Limit product info to a reasonable length
            if len(product_info) > 300:
                product_info = product_info[:297] + "..."
            summary.append(f"Products/Services: {product_info}")
        else:
            # Try to extract product info from content
            product_matches = re.search(r'(?:our|the)\s+(?:product|platform|solution|software|app|service)\s+(?:is|enables|helps|allows|provides)([^.]+)', content_to_analyze, re.IGNORECASE)
            if product_matches:
                summary.append(f"Product: {product_matches.group(1).strip()}")
    
    # Add pricing information if available
    if company_info.get('pricing'):
        pricing = company_info['pricing']
        if pricing:
            pricing_summary = []
            for tier in pricing:
                if 'name' in tier and 'price' in tier:
                    pricing_summary.append(f"• {tier['name']}: {tier['price']}")
                elif 'info' in tier:
                    pricing_summary.append(f"• {tier['info']}")
            
            if pricing_summary:
                summary.append("Pricing:")
                summary.extend(pricing_summary)
    
    # Add features
    if company_info.get('features'):
        features = company_info['features']
        if len(features) > 0:
            summary.append("Key Features:")
            for i, feature in enumerate(features[:5]):  # Limit to 5 features
                summary.append(f"• {feature}")
    
    # Add target audience
    if company_info.get('target_audience'):
        target = company_info['target_audience']
        # Clean up the target audience text
        target = re.sub(r'\s+', ' ', target).strip()
        summary.append(f"Target Audience: {target}")
    
    # Extract mission statement
    mission_matches = re.search(r'(?:our|the)\s+mission(?:\s+is)?(?:\s+to)?([^.]+)', content_to_analyze, re.IGNORECASE)
    if mission_matches:
        summary.append(f"Mission: {mission_matches.group(1).strip()}")
    
    # If we couldn't extract specific information, provide a general summary
    if len(summary) <= 2:  # Only title and description
        # Get the first 2-3 sentences from the main content
        if content_to_analyze:
            sentences = re.split(r'(?<=[.!?])\s+', content_to_analyze)
            summary_text = ' '.join(sentences[:3])
            summary.append(f"Summary: {summary_text}")
    
    return '\n'.join(summary)


def extract_features(soup):
    """Extract key features of the product or service."""
    features = []
    
    # Look for feature sections
    feature_sections = soup.select('.features, #features, .feature-list, .benefits, #benefits')
    if feature_sections:
        for section in feature_sections:
            # Look for list items in the feature section
            list_items = section.find_all('li')
            if list_items:
                for item in list_items:
                    feature_text = item.get_text(strip=True)
                    if feature_text and len(feature_text) > 10:  # Ignore very short items
                        features.append(feature_text)
    
    # Look for feature headings
    feature_headings = soup.find_all(['h2', 'h3', 'h4'], string=lambda s: s and any(keyword in s.lower() for keyword in ['feature', 'benefit', 'capability', 'advantage']))
    
    for heading in feature_headings:
        # Find the closest list after the heading
        feature_list = heading.find_next('ul')
        if feature_list:
            list_items = feature_list.find_all('li')
            for item in list_items:
                feature_text = item.get_text(strip=True)
                if feature_text and len(feature_text) > 10:  # Ignore very short items
                    features.append(feature_text)
        
        # If no list found, try to get the paragraph following the heading
        if not feature_list:
            next_element = heading.find_next(['p', 'div'])
            if next_element:
                feature_text = next_element.get_text(strip=True)
                if feature_text and len(feature_text) > 10:
                    features.append(feature_text)
    
    # Look for feature cards (common pattern in modern websites)
    feature_cards = soup.select('.card, .feature-card, .feature-box, .box')
    for card in feature_cards:
        # Try to find heading and description in the card
        card_heading = card.find(['h2', 'h3', 'h4', 'h5'])
        card_description = card.find('p')
        
        if card_heading and card_description:
            heading_text = card_heading.get_text(strip=True)
            description_text = card_description.get_text(strip=True)
            feature_text = f"{heading_text}: {description_text}"
            if len(feature_text) > 15:  # Ensure it's substantial
                features.append(feature_text)
    
    return features


def extract_target_audience(soup, content):
    """Extract information about the target audience or use cases."""
    target_audience = ""
    
    # Look for specific sections about target audience
    audience_sections = soup.select('.audience, #audience, .customers, #customers, .use-cases, #use-cases, .industries, #industries')
    if audience_sections:
        for section in audience_sections:
            text = section.get_text(strip=True)
            if text:
                target_audience += text + " "
    
    # Look for "Who it's for" or "Ideal for" sections
    audience_headings = soup.find_all(['h2', 'h3', 'h4'], string=lambda s: s and any(phrase in s.lower() for phrase in ["who it's for", "ideal for", "perfect for", "designed for", "target audience", "for whom"]))
    
    for heading in audience_headings:
        # Get the content following the heading
        next_element = heading.find_next(['p', 'div', 'ul'])
        if next_element:
            audience_text = next_element.get_text(strip=True)
            if audience_text:
                target_audience += audience_text + " "
    
    # Try to extract from content using regex patterns
    if not target_audience:
        # Look for phrases like "designed for", "perfect for", "ideal for", etc.
        audience_patterns = [
            r'(?:designed|perfect|ideal|built|made|created|intended)\s+for\s+([^.]+)',
            r'(?:helps|enables|empowers|allows)\s+([^.]+)\s+to\s+',
            r'(?:our|the)\s+(?:customers|users|clients|audience)\s+(?:are|include)\s+([^.]+)',
            r'(?:used by|trusted by|relied on by)\s+([^.]+)',
            r'(?:serving|supporting|helping)\s+([^.]+)\s+(?:since|for|to)'
        ]
        
        for pattern in audience_patterns:
            matches = re.search(pattern, content, re.IGNORECASE)
            if matches:
                target_audience = matches.group(1).strip()
                break
    
    # Look for testimonials or case studies - these often reveal target industries
    testimonials = soup.select('.testimonial, .case-study, .customer-story, .customer')
    if testimonials and not target_audience:
        industries = set()
        for testimonial in testimonials:
            # Look for company name or industry mentions
            company_text = testimonial.get_text(strip=True)
            # Extract potential industry names using common patterns
            industry_matches = re.findall(r'(?:from|at|in the)\s+(\w+(?:\s+\w+)?)\s+(?:industry|sector|field)', company_text, re.IGNORECASE)
            for match in industry_matches:
                if len(match) > 3:  # Avoid very short matches
                    industries.add(match)
        
        if industries:
            target_audience = "Industries: " + ", ".join(industries)
    
    return target_audience


def extract_product_info(soup):
    """Extract information about products or services."""
    product_info = ""
    
    # Look for product sections
    product_sections = soup.select('.product, .products, #products, .services, #services, .solutions, #solutions')
    if product_sections:
        for section in product_sections:
            # Extract text and clean it
            text = re.sub(r'\s+', ' ', section.get_text(separator=' ', strip=True))
            product_info += text + " "
    
    # Look for product headings
    product_headings = soup.find_all(['h1', 'h2', 'h3'], string=lambda s: s and any(keyword in s.lower() for keyword in ['product', 'solution', 'service', 'platform', 'software', 'app', 'application']))
    
    for heading in product_headings:
        # Get the heading text
        heading_text = heading.get_text(strip=True)
        
        # Try to get the paragraph following the heading
        next_element = heading.find_next(['p', 'div'])
        if next_element:
            paragraph_text = next_element.get_text(strip=True)
            product_info += f"{heading_text}: {paragraph_text} "
    
    # Look for pricing information
    pricing_sections = soup.select('.pricing, #pricing, .plans, #plans, .packages, #packages')
    if pricing_sections:
        for section in pricing_sections:
            pricing_text = section.get_text(strip=True)
            if pricing_text:
                product_info += f" Pricing: {pricing_text[:200]}... "
    
    return product_info.strip()


def find_about_link(soup):
    """Find a link to the About page."""
    about_keywords = ['about', 'about us', 'our company', 'who we are', 'mission']
    
    # Look for links containing about keywords
    for keyword in about_keywords:
        links = soup.find_all('a', string=lambda s: s and keyword.lower() in s.lower())
        if links:
            return links[0].get('href')
    
    # Try to find by href
    for keyword in about_keywords:
        links = soup.find_all('a', href=lambda href: href and keyword.lower() in href.lower())
        if links:
            return links[0].get('href')
    
    return None


def extract_main_content(soup):
    """Extract the main content from a webpage."""
    # Try to find main content areas
    main_content = ""
    
    # Look for common content containers
    content_containers = soup.select('main, article, .content, #content, .main, #main')
    if content_containers:
        for container in content_containers:
            # Extract text and remove excessive whitespace
            text = re.sub(r'\s+', ' ', container.get_text(separator=' ', strip=True))
            if len(text) > len(main_content):
                main_content = text
    
    # If no main content found, try to get text from the body
    if not main_content:
        body = soup.find('body')
        if body:
            # Remove script, style, and nav elements
            for element in body.select('script, style, nav, footer, header'):
                element.extract()
            
            # Get text and clean it
            main_content = re.sub(r'\s+', ' ', body.get_text(separator=' ', strip=True))
    
    return main_content