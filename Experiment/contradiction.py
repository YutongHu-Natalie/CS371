import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

def load_processed(directory_path, speaker):
    statement = []
    segment1 = []
    segment2 = []
    segment3 = []
    segment4 = []
    segment5 = []
    path_dir = Path(directory_path)

    # Check if the directory exists
    if not path_dir.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    # Get all JSON files in the directory
    files = sorted(path_dir.glob('*.json'))

    # Warn if no files are found
    if not files:
        print(f"Warning: No JSON files found in {directory_path}")
        return statement, segment1, segment2, segment3, segment4, segment5

    # Process each JSON file
    for file_path in files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            # Extract the main statement
            query_text = data.get("query_text", "")
            statement.append(query_text)
            
            # Extract matched segments for the specified speaker
            results = data.get("results", {})
            speaker_results = results.get(query_text, {}).get(speaker, [])
            for i, result in enumerate(speaker_results[:5]):  # Only take up to 5 segments
                segment = result.get("matched_segment", {}).get("text", "")
                if i == 0:
                    segment1.append(segment)
                elif i == 1:
                    segment2.append(segment)
                elif i == 2:
                    segment3.append(segment)
                elif i == 3:
                    segment4.append(segment)
                elif i == 4:
                    segment5.append(segment)
            
            # Fill missing segments with empty strings to align with the statement length
            while len(segment1) < len(statement): segment1.append("")
            while len(segment2) < len(statement): segment2.append("")
            while len(segment3) < len(statement): segment3.append("")
            while len(segment4) < len(statement): segment4.append("")
            while len(segment5) < len(statement): segment5.append("")
    
    return statement, segment1, segment2, segment3, segment4, segment5


def load_client():
    """
    Load and initialize the OpenAI client using API key from environment variables.
    """
    load_dotenv()
    key = os.getenv('OPEN_AI_API')
    if not key:
        raise ValueError("OPEN_AI_API not found in environment variables")
    
    client = OpenAI(api_key=key)
    print("OpenAI client initialized.")
    return client

def Contradiction_detection(statements, seg1, seg2, seg3, seg4, seg5, gpt_client):
    """
    Detect and classify contradictions between statements from the debate and past speeches.

    Args:
    - statements (list): Debate statements to analyze.
    - seg1, seg2, seg3, seg4, seg5 (list): Segments from past speeches.
    - gpt_client: OpenAI GPT client for text analysis.

    Returns:
    - List of dictionaries with contradiction label and explanation.
    """

    # Examples of contradictions for GPT to reference
    examples = (
        "Examples:\n"
        "1. Contradictory (Label: 1):\n"
        "   Debate: 'I have never supported policy X.'\n"
        "   Past: 'I supported policy X in 2020.'\n"
        "   Analysis: These statements represent direct logical opposites; one must be true, and the other false.\n\n"
        "2. Contrary (Label: 2):\n"
        "   Debate: 'I believe the solution is exclusively federal regulation.'\n"
        "   Past: 'I believe the solution is exclusively state control.'\n"
        "   Analysis: These are mutually exclusive claims that cannot both be true but could both be false.\n\n"
        "3. Subaltern (Label: 3):\n"
        "   Debate: 'The program will cover all workers.'\n"
        "   Past: 'The program will only cover full-time workers.'\n"
        "   Analysis: The first statement logically contains the second, creating a scope conflict.\n\n"
        "4. Numeric Mismatch (Label: 4):\n"
        "   Debate: 'The program will cost $2 billion.'\n"
        "   Past: 'I've always said this program will cost $5 billion.'\n"
        "   Analysis: There is a specific quantitative contradiction in the stated costs.\n\n"
        "5. No Contradiction (Label: 0):\n"
        "   Debate: 'We aim to improve education for all.'\n"
        "   Past: 'We have allocated funds to support teachers.'\n"
        "   Analysis: These statements are consistent and do not conflict.\n\n"
    )

    size = len(statements)
    results = []

    for i in range(size):
        debate_statement = statements[i]
        segments = [seg1[i], seg2[i], seg3[i], seg4[i], seg5[i]]
        segments = [seg for seg in segments if seg]  # Remove empty segments

        # Skip if no segments to compare
        if not segments:
            results.append({
                "Label": 0,
                "Explanation": "No past speeches available for comparison."
            })
            continue

        # Construct the input for GPT analysis
        input_prompt = (
            "You are an expert in political science tasked with detecting and classifying contradictions between "
            f"statements of a candidate from the 2024 Presidential Debate and their past speeches.\n\n"
            f"Debate Statement:\n{debate_statement}\n\n"
            "Past Speeches:\n"
            + "\n".join([f"{idx + 1}. {seg}" for idx, seg in enumerate(segments)])
            + "\n\n"
            "Your goal is to implement the detection and classification of contradiction based on the following four contradiction categories:\n"
            "1: Contradictory - Direct logical opposites where one statement must be true and the other false.\n"
            "2: Contrary - Mutually exclusive claims that cannot both be true but could both be false.\n"
            "3: Subaltern - When one statement logically contains or implies the other but they conflict (usually about scope).\n"
            "4: Numeric Mismatch - Specific quantitative contradictions in numbers, statistics, or data claims.\n"
            "0: No contradiction detected.\n\n"
            "When analyzing statements, ensure you:\n"
            "1. Identify potential contradictions between the debate statement and past speeches.\n"
            "2. Classify each contradiction using the corresponding labels.\n"
            "3. Explain the logical basis for the contradiction and its classification.\n"
            "4. Focus on substantive policy or factual contradictions.\n"
            "5. Ignore past accomplishments and future proposals on similar topics as contradictions.\n"
            "6. Ignore trivial implementation timelines, rhetoric, and procedural details unless they fundamentally alter the contradiction.\n\n"
            + examples +
            "Output format (maximum 150 tokens):\n"
            "Label: [0-4]\n"
            "Explanation: [Your concise analysis of the contradiction, if any]\n\n"
        )

        # Use GPT to analyze the contradiction
        response = gpt_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in political science analyzing contradictions."},
                {"role": "user", "content": input_prompt}
            ]
        )

        # Extract the response content
        result_text = response.choices[0].message.content
        try:
            # Parse the response into label and explanation
            label_line, explanation_line = result_text.split("\n", 1)
            label = int(label_line.replace("Label:", "").strip())
            explanation = explanation_line.replace("Explanation:", "").strip()
            results.append({"Label": label, "Explanation": explanation})
        except Exception as e:
            results.append({
                "Label": -1,
                "Explanation": f"Error processing GPT response: {str(e)}"
            })

    return results




script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
harris_path = os.path.join(root_dir, "CS371\Information Retrieval\Harris")
speaker = "Harris"
gpt_client = load_client()
statements, seg1, seg2, seg3, seg4, seg5 = load_processed(harris_path, speaker)
print(statements[0])
print("\n----------------------------------------------------------------------------------")
print(seg1[0])


# Example data
statements = ["I have always supported universal healthcare."]
seg1 = ["I opposed universal healthcare in 2019."]
seg2 = [""]
seg3 = [""]
seg4 = [""]
seg5 = [""]

results = Contradiction_detection(statements, seg1, seg2, seg3, seg4, seg5, gpt_client)

# Print results
for result in results:
    print(result)