'''
Intending to segment the debate transcript and store them into structured json file
'''
import json
import os


def process_statements(input_file):
    # Read the text file
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content into statements based on empty lines
    statements = [stmt.strip() for stmt in content.split('\n\n') if stmt.strip()]

    # Create the JSON structure
    output = {"statements": []}

    # Process each statement
    for idx, statement in enumerate(statements, 1):
        statement_obj = {
            "Statement ID": f"{str(idx-1).zfill(3)}",  # Creates IDs
            "Statement Topic": "",
            "Statement Content": statement,
            "Statement Importance": "",
            "Statement Classification": "",
            "Statement Ideology Score": ""
        }
        output["statements"].append(statement_obj)

    return output


def save_json(data, output_file):
    # Save the JSON with proper formatting
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main():
    try:
        candidate = "harris"

        # Go up two levels from current script location to reach Contenet directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        contenet_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up to root directory

        # Construct paths relative to Contenet directory
        input_file = os.path.join(contenet_dir, "Content", "Raw Materials", f"{candidate}_cleaned_statements.txt")
        output_file = os.path.join(contenet_dir, "Content", "Processed Materials", f"{candidate}_debate",f"{candidate}_debate_statements.json")

        print(input_file)

        # Process the statements
        json_data = process_statements(input_file)

        # Save to JSON file
        save_json(json_data, output_file)

        print(f"Successfully processed statements and saved to {output_file}")

    except FileNotFoundError:
        print("Error: Input file not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()