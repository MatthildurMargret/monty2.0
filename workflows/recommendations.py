import pandas as pd

user_map = {
    "Todd": "U03KZ1KQF",
    "Matt": "U9V59N8R1",
    "Daphne": "U03L4HK4S",
    "Connie": "U03JA4UARF1",
    "Nia": "U04V74ZMZ7F",
    "Matthildur": "U07MTGUFMSB"
}

def find_new_recs(username):
    from services.tree import get_nodes_and_names
    from services.database import get_db_connection
    
    conn = get_db_connection()
    if not conn:
        print("❌ Failed to connect to the database")
        return None
    try:
        new_profiles = pd.read_sql(f"""
        SELECT *
        FROM founders
        WHERE founder = true
        AND history = ''
        AND (tree_result = 'Strong recommend' OR tree_result = 'Recommend')
        """, conn)
        new_profiles = new_profiles.drop_duplicates(subset=['name'])
        new_profiles = new_profiles.reset_index(drop=True)        
    except Exception as e:
        print(f"Error fetching new profiles: {e}")
        return None
    finally:
        if conn:
            conn.close()

    columns_to_share = ['name', 'company_name', 'company_website', 'profile_url', 'tree_thesis', 'product', 'market', 'tree_path', 'past_success_indication_score', 'tree_result']
    recs = new_profiles[columns_to_share]
    recs.drop_duplicates(subset=['company_name'], inplace=True)
    recs.drop_duplicates(subset=['name'], inplace=True)
    recs.drop_duplicates(subset=['profile_url'], inplace=True)
    recs = recs.reset_index(drop=True)

    nodes_and_names = get_nodes_and_names()

    recs['top_category'] = recs['tree_path'].apply(lambda x: x.split(' > ')[0])

    for user, categories in nodes_and_names.items():
        if user != username:
            continue
        print("Profiles for ", user)
        profiles = pd.DataFrame()
        for category in categories:
            profiles = pd.concat([profiles, recs[recs['tree_path'].str.contains(category)]])
        profiles['category'] = profiles['tree_path'].apply(lambda x: x.split(' > ')[-2] if len(x.split(' > ')) > 1 else x)

        #profiles = profiles.sort_values(by=['tree_result', 'past_success_indication_score'], ascending=[False, False])
        profiles = profiles.groupby('category').head(1)

        profiles = profiles.drop_duplicates(subset=['company_name'])
        profiles = profiles.drop_duplicates(subset=['name'])
        profiles = profiles.reset_index(drop=True)
        profiles.sort_values(by=['tree_result', 'past_success_indication_score'], ascending=[False, False], inplace=True)

        for _, row in profiles.iterrows():
            print(row['tree_path'], row['company_name'], row['tree_result'], row['profile_url'])


user = "Matthildur"
find_new_recs(user)