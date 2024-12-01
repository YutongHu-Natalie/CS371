import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from anthropic import Anthropic


def load_processed(directory_path, speaker):
    statement = []
    segment1 = []
    segment2 = []
    segment3 = []
    path_dir = Path(directory_path)

    # Check if the directory exists
    if not path_dir.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Get all JSON files in the directory
    files = sorted(path_dir.glob('*.json'))

    # Warn if no files are found
    if not files:
        print(f"Warning: No JSON files found in {directory_path}")
        return statement, segment1, segment2, segment3

    # Process each JSON file
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

                # Extract the main statement (query_text)
                query_text = data.get("query_text", "")
                statement.append(query_text)

                # Extract matched segments
                # The results structure is: {"query_text": [{"matched_segment": {"text": "..."}}, ...]}
                results = data.get("results", {})
                matched_segments = results.get(query_text, [])

                # Process up to 3 segments
                current_segments = []
                for segment in matched_segments:
                    if "matched_segment" in segment:
                        segment_text = segment["matched_segment"].get("text", "")
                        current_segments.append(segment_text)
                        if len(current_segments) == 3:  # Only take first 3 segments
                            break

                # Add segments or empty strings if fewer than 3 segments
                segment1.append(current_segments[0] if len(current_segments) > 0 else "")
                segment2.append(current_segments[1] if len(current_segments) > 1 else "")
                segment3.append(current_segments[2] if len(current_segments) > 2 else "")

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            # Add empty entries to maintain alignment
            statement.append("")
            segment1.append("")
            segment2.append("")
            segment3.append("")

    return statement, segment1, segment2, segment3


def load_client(model_provider):
    """
    Load and initialize client based on the model provider.
    """
    load_dotenv()

    if model_provider == "openai":
        key = os.getenv('gpt_key')
        if not key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return OpenAI(api_key=key)

    elif model_provider == "anthropic":
        key = os.getenv('Claude_key')
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return Anthropic(api_key=key)

    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")


def get_model_response(model_name, prompt, client):
    """
    Get response from specified model with strictly formatted output.
    """
    formatted_prompt = prompt + "\n\nIMPORTANT: Your response must start with 'Label: ' followed by a number (0-4), then a new line with 'Explanation: ' followed by your analysis. Do not include any other text before these lines."

    if model_name in ["gpt-4o-mini", "gpt-4o"]:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system",
                 "content": "You are an expert in political science analyzing contradictions. Provide responses in the exact format: 'Label: [0-4]\\nExplanation: [analysis]'"},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        return response.choices[0].message.content

    elif model_name in ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"]:
        response = client.messages.create(
            model=model_name,
            system="You are an expert in political science analyzing contradictions. IMPORTANT: Your response must start with 'Label: ' followed by a number (0-4), then a new line with 'Explanation: ' followed by your analysis. Do not include any other text.",
            max_tokens=150,
            temperature=0.1,
            messages=[{"role": "user", "content": formatted_prompt}]
        )
        return response.content[0].text

    else:
        raise ValueError(f"Unsupported model: {model_name}")


def create_output_directory(base_path, speaker, model_name):
    """
    Create and return path to model-specific output directory.
    """
    dir_name = f"{speaker}_{model_name.replace('-', '_')}"
    output_dir = os.path.join(base_path, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def Contradiction_detection(statements, seg1, seg2, seg3, model_name, client, speaker, output_base_path):
    """
    Detect contradictions using the specified model and save results.
    """
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

    output_dir = create_output_directory(output_base_path, speaker, model_name)
    results = []

    for i in range(len(statements)):
        debate_statement = statements[i]
        segments = [seg1[i], seg2[i], seg3[i]]
        segments = [seg for seg in segments if seg]

        if not segments:
            results.append({
                "Label": 0,
                "Explanation": "No past speeches available for comparison."
            })
            continue

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
            + examples +
            "Provide your output in exact this format:\n"
            "Label: [0-4]\n"
            "Explanation: [Your concise analysis of the contradiction, if any]\n\n"
        )

        try:
            result_text = get_model_response(model_name, input_prompt, client)
            label_line, explanation_line = result_text.split("\n", 1)
            label = int(label_line.replace("Label:", "").strip())
            explanation = explanation_line.replace("Explanation:", "").strip()
            results.append({"Label": label, "Explanation": explanation})
        except Exception as e:
            results.append({
                "Label": -1,
                "Explanation": f"Error processing response: {str(e)}"
            })

    # Save results for this model
    output_file = os.path.join(output_dir, f"contradiction_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "model": model_name,
            "results": results
        }, f, indent=2)

    return f"Analysis completed for {model_name}"


if __name__ == "__main__":
    # Available models
    MODELS = {
        "gpt4o": "gpt-4o",
        "gpt4o-mini": "gpt-4o-mini",
        "claude-sonnet": "claude-3-5-sonnet-latest",
        "claude-haiku": "claude-3-5-haiku-latest"
    }

    # Get model selection from user
    print("Available models:")
    for key, value in MODELS.items():
        print(f"- {key}: {value}")

    model_key = input("Select a model (enter the key, e.g., 'gpt4'): ").strip()

    if model_key not in MODELS:
        raise ValueError(f"Invalid model selection. Choose from: {', '.join(MODELS.keys())}")

    model_name = MODELS[model_key]
    model_provider = "openai" if model_name.startswith("gpt") else "anthropic"
    #initialize the speaker
    speaker = "Trump"

    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    speaker_path = os.path.join(root_dir, "CS371", "Information Retrieval", speaker)
    output_base_path = os.path.join(root_dir, "CS371", "Results")

    # Initialize

    client = load_client(model_provider)
    statements, seg1, seg2, seg3 = load_processed(speaker_path, speaker)

    # Run analysis
    results = Contradiction_detection(
        statements, seg1, seg2, seg3,
        model_name, client,
        speaker, output_base_path
    )

    print(results)