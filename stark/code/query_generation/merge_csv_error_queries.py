import json
import csv
import os

# Paths
json_path = '/home/wlia0047/ar57/wenyu/result/query/generated_kg_queries_with_errors.json'
# Note: The user was viewing /home/wlia0047/ar57/wenyu/result/generated_kg_queries.csv
csv_path = '/home/wlia0047/ar57/wenyu/result/generated_kg_queries.csv'
output_path = '/home/wlia0047/ar57/wenyu/result/query/generated_kg_queries_with_errors.csv'

def main():
    print(f"Loading error queries from {json_path}...")
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        json_results = data.get('results', [])

    print(f"Loading answer_ids_source from {csv_path}...")
    if not os.path.exists(csv_path):
         print(f"Error: Original CSV file not found at {csv_path}")
         return

    id_to_answers = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Handle potential BOM or whitespace in keys
                id_key = next((k for k in row.keys() if k.strip() == 'id'), 'id')
                ans_key = next((k for k in row.keys() if k.strip() == 'answer_ids_source'), 'answer_ids_source')
                
                q_id = int(row[id_key])
                id_to_answers[q_id] = row[ans_key]
            except ValueError:
                continue

    print(f"Merging and saving to {output_path}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'query', 'answer_ids_source'])
        
        count = 0
        for item in json_results:
            q_id = item.get('query_id')
            query = item.get('error_modified_query')
            
            # Retrieve answer_ids_source, default to empty list string if missing
            answer_ids = id_to_answers.get(q_id, "[]")
            
            writer.writerow([q_id, query, answer_ids])
            count += 1
            
    print(f"✅ Successfully wrote {count} rows to {output_path}")

if __name__ == "__main__":
    main()
