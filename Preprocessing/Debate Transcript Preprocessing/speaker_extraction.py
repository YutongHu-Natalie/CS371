import re
import os


def extract_statements(file_path, speaker):
    """
    Extract statements made by a specific speaker from a debate transcript file.

    Args:
        file_path (str): Path to the debate transcript file
        speaker (str): The speaker's name to extract ("TRUMP" or "HARRIS")

    Returns:
        str: A formatted string containing all statements by the speaker
    """
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Find all instances of speaker's statements
    pattern = rf"{speaker}: (.*?)(?=\n[A-Z]+: |\Z)"
    statements = re.findall(pattern, text, re.DOTALL)

    # Initialize output as string
    output = ""

    # Add each statement
    for statement in statements:
        # Clean up the statement
        clean_statement = statement.strip()
        output += f"{clean_statement}\n\n"

    return output

# Go up two levels from current script location to reach Contenet directory
script_dir = os.path.dirname(os.path.abspath(__file__))
contenet_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up to root directory

# Construct paths relative to Contenet directory
input_path = os.path.join(contenet_dir, "Content", "Raw Materials", "debate_transcript.txt")
output_dir = os.path.join(contenet_dir, "Content", "Raw Materials")
output_trump= os.path.join(output_dir, "trump_statements.txt")
output_harris= os.path.join(output_dir, "harris_statements.txt")

print(contenet_dir)
# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Extract statements for each speaker
trump_statements = extract_statements(input_path, "TRUMP")
harris_statements = extract_statements(input_path, "HARRIS")

# Write to files
with open(output_trump, 'w') as f:
    f.write(trump_statements)

with open(output_harris, 'w') as f:
    f.write(harris_statements)

print("Files have been created successfully!")
