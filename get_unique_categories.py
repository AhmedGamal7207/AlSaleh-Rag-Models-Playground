import json
import os

def get_unique_categories(file_path):
    print(f"Reading {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    unique_categories = set()
    
    # Determine iterable based on root type
    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # Heuristic: verify if there is a main key that holds the list
        # If not, maybe the values are the items
        print("Root is a dictionary. checking keys...")
        # detailed inspection could trigger here, but let's assume standard structure or iterate values if they look like docs
        # Common patterns: {"laws": [...]} or just keys as IDs. 
        # For now, let's look for known keys or iterate values.
        for key, value in data.items():
            if isinstance(value, list):
                 items.extend(value)
            elif isinstance(value, dict):
                 items.append(value)
    
    print(f"Found {len(items)} items/sequences to process.")

    for item in items:
        if not isinstance(item, dict):
            continue
            
        categories = item.get("categories")
        
        if categories:
            if isinstance(categories, list):
                for cat in categories:
                    if isinstance(cat, str):
                        unique_categories.add(cat.strip())
            elif isinstance(categories, str):
                unique_categories.add(categories.strip())
    
    print("\nUnique Categories Found:")
    print("-" * 30)
    sorted_cats = sorted(list(unique_categories))
    for cat in sorted_cats:
        print(cat)
    print("-" * 30)
    print(f"Total unique categories: {len(unique_categories)}")

if __name__ == "__main__":
    file_path = "القوانين.json"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        get_unique_categories(file_path)
