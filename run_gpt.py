import json
import pandas as pd
import argparse
from openai import OpenAI

client = OpenAI(api_key="Your key")

def generate_responses(file_path, model):
    
    # Read the parquet file
    df = pd.read_parquet(file_path)

    l = []
 
    for i in range(len(df)):
        doc = {}
        # Generate response using the specified model
        response = client.chat.completions.create(
            model=model,
            temperature=0.7,
            max_tokens=4096,
            seed=0,
            messages=[
                {"role": "system", "content": "You are now a helpful assistant specifically designed to support ophthalmologists."},
                {"role": "user", "content": df.iloc[i]['query']}
            ]
        )
        
        # Extract the generated text and save it with the truth and query
        doc["logit_0"] = response.choices[0].message.content
        doc["truth"] = df.iloc[i]['answer']
        doc["query"] = df.iloc[i]['query']
        
        l.append(doc)
    
    # Convert the list of documents to JSON format
    json_data = json.dumps(l, indent=4)
    
    # Write the JSON data to a file
    output_file_path = file_path.replace('.parquet', '.json')
    with open(output_file_path, 'w') as file:
        file.write(json_data)

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description="Generate model responses for given queries.")
    
    parser.add_argument("file_path", type=str, help="File path to the parquet file")
    parser.add_argument("model", type=str, help="Model to use for generating responses")
    # Parse arguments
    args = parser.parse_args()
    
    # Run the function with command-line arguments
    generate_responses(args.file_path, args.model)
